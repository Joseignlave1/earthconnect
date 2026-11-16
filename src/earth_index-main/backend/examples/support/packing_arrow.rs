//! Explicit SQL packing helper: no score calculation or quantization is hidden
//! in the driver. SQL supplies ordered integer codes; this only writes bytes.
use duckdb::{
    arrow::{
        array::{Array, BinaryBuilder, Int32Array, ListArray, RecordBatch},
        datatypes::{DataType, Field},
    },
    vscalar::{ArrowFunctionSignature, VArrowScalar},
};
use std::{error::Error, sync::Arc};

pub struct PackU16;
impl VArrowScalar for PackU16 {
    type State = ();
    fn signatures() -> Vec<ArrowFunctionSignature> {
        vec![ArrowFunctionSignature::exact(
            vec![DataType::List(Arc::new(Field::new(
                "item",
                DataType::Int32,
                true,
            )))],
            DataType::Binary,
        )]
    }
    fn invoke(_: &(), input: RecordBatch) -> Result<Arc<dyn Array>, Box<dyn Error>> {
        let lists = input
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or("Expected INTEGER[]")?;
        let mut out = BinaryBuilder::new();
        for row in 0..lists.len() {
            if lists.is_null(row) {
                out.append_null();
                continue;
            }
            let values = lists.value(row);
            let codes = values
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or("Expected INTEGER[]")?;
            let mut bytes = Vec::with_capacity(codes.len() * 2);
            for code in codes.iter() {
                let code = match code {
                    None => u16::MAX,
                    Some(v) => {
                        u16::try_from(v).map_err(|_| "pack_u16 accepts only 0..65535 or NULL")?
                    }
                };
                bytes.extend_from_slice(&code.to_le_bytes());
            }
            out.append_value(bytes);
        }
        Ok(Arc::new(out.finish()))
    }
}

/// Scatter explicit zero-based addresses instead of sorting millions of cells.
/// Missing addresses default to 65535; duplicates/out-of-range codes are errors.
pub struct PackU16At;
impl VArrowScalar for PackU16At {
    type State = ();
    fn signatures() -> Vec<ArrowFunctionSignature> {
        vec![ArrowFunctionSignature::exact(
            vec![
                DataType::List(Arc::new(Field::new(
                    "item",
                    DataType::Struct(
                        vec![
                            Field::new("index", DataType::UInt32, true),
                            Field::new("code", DataType::Int32, true),
                        ]
                        .into(),
                    ),
                    true,
                ))),
                DataType::UInt32,
            ],
            DataType::Binary,
        )]
    }
    fn invoke(_: &(), input: RecordBatch) -> Result<Arc<dyn Array>, Box<dyn Error>> {
        use duckdb::arrow::array::{StructArray, UInt32Array};
        let lists = input
            .column(0)
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or("Expected addressed list")?;
        let lengths = input
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or("Expected matrix length")?;
        let mut out = BinaryBuilder::new();
        for row in 0..lists.len() {
            if lists.is_null(row) || lengths.is_null(row) {
                out.append_null();
                continue;
            }
            let length = lengths.value(row) as usize;
            if length == 0 || length > 5_000_000 {
                return Err("Matrix length must be 1..5000000".into());
            }
            let values = lists.value(row);
            let entries = values
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or("Expected addressed entries")?;
            let indices = entries
                .column(0)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or("Expected UINTEGER index")?;
            let codes = entries
                .column(1)
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or("Expected INTEGER code")?;
            let mut bytes = vec![255; length * 2];
            let mut seen = vec![false; length];
            for i in 0..entries.len() {
                if entries.is_null(i) || indices.is_null(i) {
                    return Err("Matrix index cannot be NULL".into());
                }
                let index = indices.value(i) as usize;
                if index >= length || seen[index] {
                    return Err("Matrix index out of range or duplicated".into());
                }
                seen[index] = true;
                let code = if codes.is_null(i) {
                    u16::MAX
                } else {
                    u16::try_from(codes.value(i))
                        .map_err(|_| "Score code must be 0..65535 or NULL")?
                };
                bytes[index * 2..index * 2 + 2].copy_from_slice(&code.to_le_bytes());
            }
            out.append_value(bytes);
        }
        Ok(Arc::new(out.finish()))
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
}
