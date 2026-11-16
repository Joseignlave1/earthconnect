//! Read-only release benchmark. Uses the same SQL, packing and row/protobuf
//! representation as production; no server or database configuration is changed.
#[path = "support/packing_arrow.rs"]
mod arrow_baseline;
#[path = "../src/packing.rs"]
mod packing;
use duckdb::{
    AccessMode, Config, Connection,
    core::DataChunkHandle,
    vscalar::{ScalarFunctionSignature, VScalar},
    vtab::arrow::WritableVector,
};
use earth_index_api::{
    params,
    proto::{self, value::Kind},
    sql, value,
};
use prost::Message;
use serde_json::json;
use std::{
    error::Error,
    sync::{Arc, Mutex},
    time::Instant,
};

#[derive(Clone, Default)]
struct Stats(Arc<Mutex<Vec<(Instant, Instant)>>>);
struct ProfiledPack;
impl VScalar for ProfiledPack {
    type State = Stats;
    fn signatures() -> Vec<ScalarFunctionSignature> {
        <packing::PackU16At as VScalar>::signatures()
    }
    fn invoke(
        stats: &Stats,
        input: &mut DataChunkHandle,
        output: &mut dyn WritableVector,
    ) -> Result<(), Box<dyn Error>> {
        let start = Instant::now();
        let result = <packing::PackU16At as VScalar>::invoke(&(), input, output);
        stats.0.lock().unwrap().push((start, Instant::now()));
        result
    }
}
type MatrixRows = Vec<(String, Vec<u8>)>;
fn arrow_control(
    c: &Connection,
    sql: &str,
    parameters: &[duckdb::types::Value],
) -> duckdb::Result<(f64, MatrixRows)> {
    let mut stmt = c.prepare(sql)?;
    let start = Instant::now();
    let mut rows = stmt.query(duckdb::params_from_iter(parameters.iter()))?;
    let execution_ms = ms(start.elapsed());
    let mut result = vec![];
    while let Some(row) = rows.next()? {
        result.push((row.get(0)?, row.get(1)?));
    }
    Ok((execution_ms, result))
}
fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.
}
fn profile_json(p: duckdb::profiling::ProfilingInfo) -> serde_json::Value {
    json!({"metrics":p.metrics,"children":p.children.into_iter().map(profile_json).collect::<Vec<_>>()})
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let data = args
        .get(1)
        .map(String::as_str)
        .unwrap_or("db/runtime/earth.duckdb");
    let extension = args
        .get(2)
        .map(String::as_str)
        .unwrap_or("db/runtime/spatial.duckdb_extension");
    let dir = args
        .get(3)
        .map(String::as_str)
        .unwrap_or("research/world-4km/pipeline-profile");
    let request =
        proto::QueryRequest::decode(std::fs::read(format!("{dir}/request.pb"))?.as_slice())?;
    let named = sql::rewrite(&request.query, &request.params).map_err(|e| format!("{e:?}"))?;
    let parameters = named
        .names
        .iter()
        .map(|n| params::bind(request.params[n].clone()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{e:?}"))?;
    let c = Connection::open_with_flags(
        data,
        Config::default()
            .access_mode(AccessMode::ReadOnly)?
            .threads(8)?
            .max_memory("8GB")?
            .enable_autoload_extension(false)?,
    )?;
    c.execute_batch(&format!(
        "LOAD '{}';SET enable_external_access=false;",
        extension.replace('\'', "''")
    ))?;
    c.register_scalar_function::<packing::PackU16>("pack_u16")?;
    let stats = Stats::default();
    c.register_scalar_function_with_state::<ProfiledPack>("pack_u16_at", &stats)?;
    c.register_scalar_function::<arrow_baseline::PackU16>("pack_u16_arrow")?;
    c.register_scalar_function::<arrow_baseline::PackU16At>("pack_u16_at_arrow")?;
    let prefix = named.sql.split("), matrices AS (").next().unwrap();
    let checksum = format!(
        "{prefix}) SELECT sum(CASE WHEN score IS NULL THEN 65535 ELSE round_even(65534*greatest(0.,least(1.,score)),0)::INTEGER END) FROM scored"
    );
    let arrow_query = named.sql.replace("pack_u16_at(", "pack_u16_at_arrow(");
    let mut samples = vec![];
    for run in 0..4 {
        let start = Instant::now();
        let mut control = c.prepare(&checksum)?;
        let sum: duckdb::types::Value =
            control.query_row(duckdb::params_from_iter(parameters.clone()), |r| r.get(0))?;
        let checksum_ms = ms(start.elapsed());
        drop(control);
        // Alternate execution order within warm paired samples.
        let before = if run % 2 == 0 {
            Some(arrow_control(&c, &arrow_query, &parameters)?)
        } else {
            None
        };
        stats.0.lock().unwrap().clear();
        let start = Instant::now();
        let mut stmt = c.prepare(&named.sql)?;
        let prepared = Instant::now();
        let mut rows = stmt.query(duckdb::params_from_iter(parameters.clone()))?;
        let executed = Instant::now();
        let mut response = proto::QueryResponse {
            columns: vec![
                proto::Column {
                    name: "site_key".into(),
                    pg_type: "Utf8".into(),
                },
                proto::Column {
                    name: "scores".into(),
                    pg_type: "Binary".into(),
                },
            ],
            rows: vec![],
        };
        let mut payload = 0;
        let mut encoded_size = 0;
        while let Some(row) = rows.next()? {
            let key: String = row.get(0)?;
            let bytes: Vec<u8> = row.get(1)?;
            payload += bytes.len();
            let row = proto::Row {
                values: vec![value::string(key), value::wrap(Kind::BytesValue(bytes))],
            };
            encoded_size += row.encoded_len() + 10;
            response.rows.push(row);
        }
        let copied = Instant::now();
        let wire = response.encode_to_vec();
        let encoded = Instant::now();
        assert_eq!(payload, 71_101_280);
        assert!(encoded_size < 134_217_728);
        let calls = stats.0.lock().unwrap().clone();
        let packing_sum_ms = calls.iter().map(|(a, b)| ms(*b - *a)).sum::<f64>();
        let packing_span_ms = ms(calls
            .iter()
            .map(|(_, b)| b)
            .max()
            .unwrap()
            .duration_since(*calls.iter().map(|(a, _)| a).min().unwrap()));
        let arrow = match before {
            Some(result) => result,
            None => arrow_control(&c, &arrow_query, &parameters)?,
        };
        for (row, (key, bytes)) in response.rows.iter().zip(&arrow.1) {
            assert_eq!(row.values[0], value::string(key));
            assert_eq!(
                row.values[1].kind.as_ref(),
                Some(&Kind::BytesValue(bytes.clone()))
            );
        }
        assert_eq!(response.rows.len(), arrow.1.len());
        let sample = json!({"run":run,"warmup":run==0,"score_checksum_ms":checksum_ms,"arrow_sql_execute_ms":arrow.0,"prepare_ms":ms(prepared-start),"sql_execute_with_packing_ms":ms(executed-prepared),"rows_to_proto_ms":ms(copied-executed),"protobuf_encode_ms":ms(encoded-copied),"prepare_to_encoded_ms":ms(encoded-start),"packing_body_summed_ms":packing_sum_ms,"packing_body_span_ms":packing_span_ms,"packing_calls":calls.len(),"score_bytes":payload,"protobuf_bytes":wire.len(),"checksum":format!("{sum:?}"),"byte_exact_arrow":true});
        println!("{sample}");
        samples.push(sample);
        drop(rows);
        drop(stmt);
    }
    c.execute_batch("PRAGMA enable_profiling='no_output'")?;
    let mut stmt = c.prepare(&named.sql)?;
    let _rows = stmt
        .query_map(duckdb::params_from_iter(parameters), |r| {
            r.get::<_, Vec<u8>>(1)
        })?
        .collect::<duckdb::Result<Vec<_>>>()?;
    let profile = profile_json(c.get_profiling_info().unwrap());
    std::fs::write(
        format!("{dir}/rust-native.json"),
        serde_json::to_vec_pretty(
            &json!({"method":"Native vector packer versus the frozen previous Arrow packer, identical 8-thread read-only file-backed Valles SQL. First paired run is warmup, then three pairs with alternating order. Every output byte compared across all 460 regions. Native packing callback time includes direct vector reads and output BLOB insertion. Scalar checksum control is separate and non-additive. Final operator profile is a separate native run.","samples":samples,"operator_profile":profile}),
        )?,
    )?;
    Ok(())
}
