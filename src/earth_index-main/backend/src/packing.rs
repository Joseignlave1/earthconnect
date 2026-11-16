//! SQL packing helpers over DuckDB's native vectors. The C scalar callback
//! recursively flattens inputs before invoking us (DuckDB 2.0 CAPIScalarFunction).
//! Read only valid, in-range slots; NULL payloads may be uninitialized. Output
//! BLOB insertion copies the bytes before the scratch buffer is reused.
use duckdb::{
    core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId},
    ffi,
    vscalar::{ScalarFunctionSignature, VScalar},
    vtab::arrow::WritableVector,
};
use std::{error::Error, marker::PhantomData, ops::Range};

type PackResult = Result<(), Box<dyn Error>>;
const MAX_MATRIX_CELLS: usize = 5_000_000;

/// Cached validity bits borrowed from one callback's flattened input. No copied
/// row-index arrays or per-value FFI calls are needed.
struct Validity<'a> {
    bits: *const u64,
    len: usize,
    _input: PhantomData<&'a DataChunkHandle>,
}
impl<'a> Validity<'a> {
    /// # Safety
    /// `vector` must be a live flattened vector in `input`, with at least `len`
    /// rows. Inputs remain unmodified throughout this callback.
    unsafe fn new(vector: ffi::duckdb_vector, len: usize, _: &'a DataChunkHandle) -> Self {
        Self {
            // SAFETY: caller guarantees a live vector for the entire borrow.
            bits: unsafe { ffi::duckdb_vector_get_validity(vector) },
            len,
            _input: PhantomData,
        }
    }
    fn valid(&self, row: usize) -> bool {
        assert!(row < self.len, "DuckDB input row out of bounds");
        // A null validity pointer means every row is valid. The bounds check
        // also covers the corresponding bitset word. We only read input bits.
        self.bits.is_null() || unsafe { *self.bits.add(row / 64) & (1_u64 << (row % 64)) != 0 }
    }
}
struct InputVector<'a, T> {
    data: *const T,
    validity: Validity<'a>,
}
impl<'a, T: Copy> InputVector<'a, T> {
    /// # Safety
    /// Same contract as Validity::new; T must match the signature's physical
    /// storage type. Valid slots are initialized, even when adjacent NULL
    /// slots are not. Never construct a typed slice spanning those NULLs.
    unsafe fn new(vector: ffi::duckdb_vector, len: usize, input: &'a DataChunkHandle) -> Self {
        Self {
            // SAFETY: type and lifetime are guaranteed by the caller.
            data: unsafe { ffi::duckdb_vector_get_data(vector).cast() },
            validity: unsafe { Validity::new(vector, len, input) },
        }
    }
    fn get(&self, row: usize) -> Option<T> {
        self.validity.valid(row).then(|| {
            // SAFETY: validity.valid checked bounds and initialization. No
            // mutable reference to input storage is ever created or retained.
            unsafe { self.data.add(row).read() }
        })
    }
}
fn list_range(
    entry: ffi::duckdb_list_entry,
    child_len: usize,
) -> Result<Range<usize>, Box<dyn Error>> {
    let start = usize::try_from(entry.offset)?;
    let len = usize::try_from(entry.length)?;
    let end = start
        .checked_add(len)
        .filter(|end| *end <= child_len)
        .ok_or("Invalid DuckDB list span")?;
    Ok(start..end)
}
fn code(value: Option<i32>) -> Result<u16, Box<dyn Error>> {
    match value {
        None => Ok(u16::MAX),
        Some(value) => {
            u16::try_from(value).map_err(|_| "Score code must be 0..65535 or NULL".into())
        }
    }
}

pub struct PackU16;
impl VScalar for PackU16 {
    type State = ();
    fn signatures() -> Vec<ScalarFunctionSignature> {
        vec![ScalarFunctionSignature::exact(
            vec![LogicalTypeHandle::list(&LogicalTypeId::Integer.into())],
            LogicalTypeId::Blob.into(),
        )]
    }
    fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> PackResult {
        let rows = input.len();
        // SAFETY: the declared INTEGER[] signature and DuckDB's recursive
        // callback flattening guarantee the list and child physical layouts.
        let (lists, values) = unsafe {
            let list = ffi::duckdb_data_chunk_get_vector(input.get_ptr(), 0);
            let child = ffi::duckdb_list_vector_get_child(list);
            let child_len = usize::try_from(ffi::duckdb_list_vector_get_size(list))?;
            (
                InputVector::<ffi::duckdb_list_entry>::new(list, rows, input),
                InputVector::<i32>::new(child, child_len, input),
            )
        };
        let mut out = output.flat_vector();
        let mut bytes = Vec::new();
        for row in 0..rows {
            let Some(entry) = lists.get(row) else {
                out.set_null(row);
                continue;
            };
            let range = list_range(entry, values.validity.len)?;
            bytes.clear();
            bytes.try_reserve(
                range
                    .len()
                    .checked_mul(2)
                    .ok_or("Packed array size overflow")?,
            )?;
            for i in range {
                bytes.extend_from_slice(&code(values.get(i))?.to_le_bytes());
            }
            out.insert(row, bytes.as_slice());
        }
        Ok(())
    }
}

/// Scatter zero-based addresses without sorting; preserves the existing SQL
/// signature, NULL sentinel, duplicate checks and little-endian wire format.
pub struct PackU16At;
impl VScalar for PackU16At {
    type State = ();
    fn signatures() -> Vec<ScalarFunctionSignature> {
        let entry = LogicalTypeHandle::struct_type(&[
            ("index", LogicalTypeId::UInteger.into()),
            ("code", LogicalTypeId::Integer.into()),
        ]);
        vec![ScalarFunctionSignature::exact(
            vec![
                LogicalTypeHandle::list(&entry),
                LogicalTypeId::UInteger.into(),
            ],
            LogicalTypeId::Blob.into(),
        )]
    }
    fn invoke(_: &(), input: &mut DataChunkHandle, output: &mut dyn WritableVector) -> PackResult {
        let rows = input.len();
        // SAFETY: DuckDB's C scalar entry recursively flattens LIST/STRUCT
        // children. The bound signature fixes all physical types. The list
        // child size covers the struct, both fields and their validity masks.
        // Every raw borrow is tied to this callback and used only for reads.
        let (lists, lengths, entries, indices, codes) = unsafe {
            let list = ffi::duckdb_data_chunk_get_vector(input.get_ptr(), 0);
            let lengths = ffi::duckdb_data_chunk_get_vector(input.get_ptr(), 1);
            let child = ffi::duckdb_list_vector_get_child(list);
            let child_len = usize::try_from(ffi::duckdb_list_vector_get_size(list))?;
            (
                InputVector::<ffi::duckdb_list_entry>::new(list, rows, input),
                InputVector::<u32>::new(lengths, rows, input),
                Validity::new(child, child_len, input),
                InputVector::<u32>::new(
                    ffi::duckdb_struct_vector_get_child(child, 0),
                    child_len,
                    input,
                ),
                InputVector::<i32>::new(
                    ffi::duckdb_struct_vector_get_child(child, 1),
                    child_len,
                    input,
                ),
            )
        };
        let mut out = output.flat_vector();
        let mut bytes = Vec::new();
        let mut seen = Vec::new();
        for row in 0..rows {
            let (Some(entry), Some(length)) = (lists.get(row), lengths.get(row)) else {
                out.set_null(row);
                continue;
            };
            let length = length as usize;
            if !(1..=MAX_MATRIX_CELLS).contains(&length) {
                return Err("Matrix length must be 1..5000000".into());
            }
            let range = list_range(entry, entries.len)?;
            bytes.resize(length * 2, 255);
            bytes.fill(255);
            seen.resize(length, false);
            seen.fill(false);
            for i in range {
                if !entries.valid(i) {
                    return Err("Matrix index cannot be NULL".into());
                }
                let index = indices.get(i).ok_or("Matrix index cannot be NULL")? as usize;
                if index >= length || seen[index] {
                    return Err("Matrix index out of range or duplicated".into());
                }
                seen[index] = true;
                bytes[index * 2..index * 2 + 2].copy_from_slice(&code(codes.get(i))?.to_le_bytes());
            }
            out.insert(row, bytes.as_slice());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use duckdb::{Config, Connection};
    fn connection() -> Connection {
        let c =
            Connection::open_in_memory_with_flags(Config::default().threads(2).unwrap()).unwrap();
        c.register_scalar_function::<PackU16>("pack_u16").unwrap();
        c.register_scalar_function::<PackU16At>("pack_u16_at")
            .unwrap();
        c
    }
    #[test]
    fn explicit_addresses_and_quantization_preserve_zero_missing_and_endpoints() {
        let c = connection();
        let blob:Vec<u8>=c.query_row("SELECT pack_u16_at(list(struct_pack(index:=i,code:=code)),5::UINTEGER) FROM (VALUES (3::UINTEGER,NULL::INTEGER),(0,0),(2,65534),(1,round_even(.5,0)::INTEGER)) t(i,code)",[],|r|r.get(0)).unwrap();
        assert_eq!(blob, vec![0, 0, 0, 0, 254, 255, 255, 255, 255, 255]);
        let blob: Vec<u8> = c
            .query_row("SELECT pack_u16([0,256,NULL,65535])", [], |r| r.get(0))
            .unwrap();
        assert_eq!(blob, vec![0, 0, 0, 1, 255, 255, 255, 255]);
    }
    #[test]
    fn malformed_matrices_fail_without_breaking_later_queries() {
        let c = connection();
        for q in [
            "SELECT pack_u16([-1])",
            "SELECT pack_u16([65536])",
            "SELECT pack_u16_at([{index:0::UINTEGER,code:1},{index:0::UINTEGER,code:2}],2::UINTEGER)",
            "SELECT pack_u16_at([{index:2::UINTEGER,code:1}],2::UINTEGER)",
            "SELECT pack_u16_at([{index:0::UINTEGER,code:65536}],1::UINTEGER)",
            "SELECT pack_u16_at([{index:0::UINTEGER,code:1}],5000001::UINTEGER)",
        ] {
            assert!(
                c.query_row(q, [], |r| r.get::<_, Vec<u8>>(0)).is_err(),
                "{q}"
            );
        }
        assert_eq!(
            c.query_row("SELECT 1", [], |r| r.get::<_, i32>(0)).unwrap(),
            1
        );
    }
    #[test]
    fn null_empty_sparse_and_sliced_lists_preserve_the_contract() {
        let c = connection();
        let row=c.query_row("SELECT pack_u16(NULL::INTEGER[]),pack_u16([]::INTEGER[]),pack_u16_at(NULL::STRUCT(index UINTEGER,code INTEGER)[],3::UINTEGER),pack_u16_at([]::STRUCT(index UINTEGER,code INTEGER)[],NULL::UINTEGER),pack_u16_at([]::STRUCT(index UINTEGER,code INTEGER)[],3::UINTEGER)",[],|r|Ok((r.get::<_,Option<Vec<u8>>>(0)?,r.get::<_,Vec<u8>>(1)?,r.get::<_,Option<Vec<u8>>>(2)?,r.get::<_,Option<Vec<u8>>>(3)?,r.get::<_,Vec<u8>>(4)?))).unwrap();
        assert_eq!(row, (None, vec![], None, None, vec![255; 6]));
        // More rows than a standard vector; filtering produces selected input
        // rows and each list's numeric children vary independently.
        let mut stmt=c.prepare("SELECT i,pack_u16_at([{index:0::UINTEGER,code:i::INTEGER},{index:2::UINTEGER,code:NULL::INTEGER}],3::UINTEGER) FROM range(6000) t(i) WHERE i%7<3 ORDER BY i").unwrap();
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)))
            .unwrap();
        let mut count = 0;
        for row in rows {
            let (i, bytes) = row.unwrap();
            let mut expected = (i as u16).to_le_bytes().to_vec();
            expected.extend_from_slice(&[255; 4]);
            assert_eq!(bytes, expected);
            count += 1;
        }
        assert!(count > 2048);
        let blob: Vec<u8> = c
            .query_row(
                "SELECT pack_u16(list_slice([99,NULL,256,65535,1],2,4))",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(blob, vec![255, 255, 0, 1, 255, 255]);
        for query in [
            "SELECT pack_u16_at([NULL::STRUCT(index UINTEGER,code INTEGER)],1::UINTEGER)",
            "SELECT pack_u16_at([{index:NULL::UINTEGER,code:0}],1::UINTEGER)",
            "SELECT pack_u16_at([{index:0::UINTEGER,code:-1}],1::UINTEGER)",
            "SELECT pack_u16_at([]::STRUCT(index UINTEGER,code INTEGER)[],0::UINTEGER)",
        ] {
            assert!(
                c.query_row(query, [], |r| r.get::<_, Vec<u8>>(0)).is_err(),
                "{query}"
            );
        }
    }

    #[test]
    fn grouped_lists_cross_vector_boundaries_and_reuse_nonzero_offsets() {
        let c = connection();
        let mut stmt=c.prepare("SELECT i//257 AS g,pack_u16_at(list(struct_pack(index:=(i%257)::UINTEGER,code:=CASE WHEN i%11=0 THEN NULL ELSE i::INTEGER END) ORDER BY i DESC),257::UINTEGER) FROM range(4369) t(i) GROUP BY g ORDER BY g").unwrap();
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)))
            .unwrap();
        for row in rows {
            let (g, bytes) = row.unwrap();
            let expected: Vec<u8> = (g * 257..(g + 1) * 257)
                .flat_map(|i| if i % 11 == 0 { u16::MAX } else { i as u16 }.to_le_bytes())
                .collect();
            assert_eq!(bytes, expected);
        }
        let rows: Vec<Option<Vec<u8>>>=c.prepare("SELECT pack_u16(CASE WHEN i%4096<2048 THEN NULL ELSE [i::INTEGER,NULL] END) FROM range(10000) t(i)").unwrap().query_map([],|r|r.get(0)).unwrap().collect::<duckdb::Result<_>>().unwrap();
        for (i, row) in rows.into_iter().enumerate() {
            if i % 4096 < 2048 {
                assert!(row.is_none());
            } else {
                let mut expected = (i as u16).to_le_bytes().to_vec();
                expected.extend_from_slice(&[255; 2]);
                assert_eq!(row, Some(expected));
            }
        }
    }
}
