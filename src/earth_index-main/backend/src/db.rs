//! Embedded DuckDB. Each request owns a fresh connection; cancellation can never
//! interrupt a later request. The shared engine has a bounded worker/memory pool.
use crate::{
    AppState,
    error::ApiError,
    params,
    proto::{self, value::Kind},
    sql, value,
};
use duckdb::{AccessMode, Config, Connection, InterruptHandle, types::Value as D};
use prost::Message;
use std::{
    path::Path,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::sync::Semaphore;
use tonic::Code;

pub struct Database {
    root: Mutex<Connection>,
    pub metadata: proto::MetadataResponse,
    pub(crate) permits: Arc<Semaphore>,
}
impl Database {
    pub fn open(
        path: impl AsRef<Path>,
        spatial: impl AsRef<Path>,
        threads: i64,
        memory: &str,
    ) -> Result<Arc<Self>, Box<dyn std::error::Error>> {
        let config = Config::default()
            .access_mode(AccessMode::ReadOnly)?
            .threads(threads)?
            .max_memory(memory)?
            .enable_autoload_extension(false)?
            .with("allow_community_extensions", "false")?
            .with("allow_persistent_secrets", "false")?;
        let conn = Connection::open_with_flags(path, config)?;
        let extension = spatial.as_ref().canonicalize()?;
        conn.execute_batch(&format!(
            "LOAD '{}'; SET enable_external_access=false; SET lock_configuration=true;",
            extension.to_string_lossy().replace('\'', "''")
        ))?;
        conn.register_scalar_function::<crate::packing::PackU16>("pack_u16")?;
        conn.register_scalar_function::<crate::packing::PackU16At>("pack_u16_at")?;
        let blob: Vec<u8> = conn.query_row("SELECT payload FROM api_metadata", [], |r| r.get(0))?;
        let mut metadata = proto::MetadataResponse::decode(blob.as_slice())?;
        crate::metadata::describe_duckdb(&mut metadata, &conn)?;
        Ok(Arc::new(Self {
            root: Mutex::new(conn),
            metadata,
            permits: Arc::new(Semaphore::new(2)),
        }))
    }
    #[cfg(test)]
    pub(crate) fn test_database() -> Arc<Self> {
        Arc::new(Self {
            root: Mutex::new(Connection::open_in_memory().unwrap()),
            metadata: proto::MetadataResponse {
                datasources: vec![proto::Datasource {
                    id: "test".into(),
                    ..Default::default()
                }],
            },
            permits: Arc::new(Semaphore::new(2)),
        })
    }
    fn connect(&self) -> Result<Connection, ApiError> {
        self.root
            .lock()
            .map_err(|_| ApiError::unavailable())?
            .try_clone()
            .map_err(Into::into)
    }
}

#[derive(Default)]
struct Cancellation {
    cancelled: AtomicBool,
    finished: AtomicBool,
    handle: Mutex<Option<Arc<InterruptHandle>>>,
}
impl Cancellation {
    fn cancel(self: &Arc<Self>) {
        if self.finished.load(Ordering::SeqCst) {
            return;
        }
        self.cancelled.store(true, Ordering::SeqCst);
        let handle = self.handle.lock().unwrap().clone();
        if let Some(handle) = handle {
            handle.interrupt();
            // DuckDB can clear an interrupt between PREPARE/BEGIN/EXECUTE.
            // Keep cancellation asserted until this private connection exits.
            let cancellation = self.clone();
            tokio::spawn(async move {
                while !cancellation.finished.load(Ordering::SeqCst) {
                    handle.interrupt();
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            });
        }
    }
    fn attach(&self, conn: &Connection) -> Result<(), ApiError> {
        let mut handle = self.handle.lock().unwrap();
        if self.cancelled.load(Ordering::SeqCst) {
            return Err(ApiError::new(
                Code::Cancelled,
                "QUERY_CANCELLED",
                "Query cancelled",
            ));
        }
        *handle = Some(conn.interrupt_handle());
        Ok(())
    }
}
struct WorkerDone(Arc<Cancellation>);
impl Drop for WorkerDone {
    fn drop(&mut self) {
        self.0.finished.store(true, Ordering::SeqCst);
    }
}
struct CancelOnDrop(Arc<Cancellation>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel();
    }
}

pub async fn execute(
    state: &AppState,
    id: &str,
    mut request: proto::QueryRequest,
) -> Result<proto::QueryResponse, ApiError> {
    if !state
        .database
        .metadata
        .datasources
        .iter()
        .any(|d| d.id == id)
    {
        return Err(ApiError::new(
            Code::NotFound,
            "DATASOURCE_NOT_FOUND",
            "Unknown datasource; call Metadata",
        ));
    }
    let named = sql::rewrite(&request.query, &request.params)?;
    let parameters = named
        .names
        .iter()
        .map(|n| params::bind(request.params.remove(n).expect("validated binding")))
        .collect::<Result<Vec<_>, _>>()?;
    let cancel = Arc::new(Cancellation::default());
    let _guard = CancelOnDrop(cancel.clone());
    let database = state.database.clone();
    let max_bytes = state.max_response_bytes;
    let operation = async {
        let permit = database
            .permits
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| ApiError::unavailable())?;
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let _done = WorkerDone(cancel.clone());
            let conn = database.connect()?;
            cancel.attach(&conn)?;
            conn.execute_batch("BEGIN TRANSACTION READ ONLY")?;
            let result = query(&conn, &named.sql, parameters, max_bytes);
            let _ = conn.execute_batch("ROLLBACK");
            result
        })
        .await
        .map_err(|_| ApiError::new(Code::Internal, "QUERY_WORKER_FAILED", "Query worker failed"))?
    };
    if state.statement_timeout_ms == 0 {
        operation.await
    } else {
        tokio::time::timeout(
            std::time::Duration::from_millis(state.statement_timeout_ms),
            operation,
        )
        .await
        .map_err(|_| {
            ApiError::new(
                Code::DeadlineExceeded,
                "QUERY_TIMEOUT",
                "Query execution deadline exceeded",
            )
        })?
    }
}

fn query(
    conn: &Connection,
    sql: &str,
    parameters: Vec<D>,
    max_bytes: usize,
) -> Result<proto::QueryResponse, ApiError> {
    let mut stmt = conn.prepare(sql)?;
    let mut rows = stmt.query(duckdb::params_from_iter(parameters))?;
    let mut response = proto::QueryResponse {
        columns: rows
            .as_ref()
            .unwrap()
            .column_names()
            .into_iter()
            .enumerate()
            .map(|(i, name)| proto::Column {
                name,
                pg_type: format!("{}", rows.as_ref().unwrap().column_type(i)),
            })
            .collect(),
        rows: vec![],
    };
    let json_columns: Vec<bool> = (0..response.columns.len())
        .map(|i| {
            rows.as_ref()
                .unwrap()
                .column_logical_type(i)
                .get_alias()
                .as_deref()
                == Some("JSON")
        })
        .collect();
    let mut size = response.encoded_len();
    while let Some(row) = rows.next()? {
        let values = (0..response.columns.len())
            .map(|i| {
                let v = row.get::<_, D>(i)?;
                if json_columns[i]
                    && let D::Text(text) = v
                {
                    return serde_json::from_str(&text)
                        .map(value::from_json)
                        .map_err(|_| ApiError::bad_request("Invalid JSON result"));
                }
                decode(v)
            })
            .collect::<Result<_, ApiError>>()?;
        let row = proto::Row { values };
        size += row.encoded_len() + 10;
        if size > max_bytes {
            return Err(ApiError::new(
                Code::ResourceExhausted,
                "RESULT_TOO_LARGE",
                "Result exceeds MAX_RESPONSE_BYTES; reduce it in SQL or use pack_u16 for score arrays",
            ));
        }
        response.rows.push(row);
    }
    Ok(response)
}
fn integer(n: i128) -> proto::Value {
    match i64::try_from(n) {
        Ok(n) => value::wrap(Kind::IntegerValue(n)),
        Err(_) => value::string(n.to_string()),
    }
}
fn decode(v: D) -> Result<proto::Value, ApiError> {
    Ok(match v {
        D::Null => value::null(),
        D::Boolean(v) => value::wrap(Kind::BoolValue(v)),
        D::TinyInt(v) => integer(v.into()),
        D::SmallInt(v) => integer(v.into()),
        D::Int(v) => integer(v.into()),
        D::BigInt(v) => integer(v.into()),
        D::HugeInt(v) => integer(v),
        D::UTinyInt(v) => integer(v.into()),
        D::USmallInt(v) => integer(v.into()),
        D::UInt(v) => integer(v.into()),
        D::UBigInt(v) => integer(v.into()),
        D::UHugeInt(v) => {
            if let Ok(n) = i64::try_from(v) {
                integer(n.into())
            } else {
                value::string(v.to_string())
            }
        }
        D::Double(v) => value::wrap(Kind::NumberValue(v)),
        D::Float(v) => value::wrap(Kind::NumberValue(v.into())),
        D::Text(v) | D::Enum(v) => value::string(v),
        D::Blob(v) | D::Geometry(v) => value::wrap(Kind::BytesValue(v)),
        D::List(v) | D::Array(v) => {
            value::list(v.into_iter().map(decode).collect::<Result<_, _>>()?)
        }
        D::Struct(v) => value::wrap(Kind::ObjectValue(proto::Object {
            fields: v
                .iter()
                .map(|(k, v)| Ok((k.clone(), decode(v.clone())?)))
                .collect::<Result<_, ApiError>>()?,
        })),
        D::Map(v) => value::list(
            v.iter()
                .map(|(k, v)| Ok(value::list(vec![decode(k.clone())?, decode(v.clone())?])))
                .collect::<Result<_, ApiError>>()?,
        ),
        D::Union(v) => decode(*v)?,
        D::Decimal(v) => value::string(v.to_string()),
        D::Date32(v) => value::string(
            chrono::NaiveDate::from_ymd_opt(1970, 1, 1)
                .unwrap()
                .checked_add_signed(chrono::Duration::days(v.into()))
                .map(|d| d.to_string())
                .unwrap_or_else(|| v.to_string()),
        ),
        D::Timestamp(unit, v) => value::string(format!("{v} {unit:?} since 1970-01-01 UTC")),
        D::Time64(unit, v) => value::string(format!("{v} {unit:?} since midnight")),
        D::Interval {
            months,
            days,
            nanos,
        } => value::string(format!("{months} months {days} days {nanos} nanoseconds")),
        _ => {
            return Err(ApiError::bad_request(
                "Unsupported result type; cast it in SQL",
            ));
        }
    })
}
