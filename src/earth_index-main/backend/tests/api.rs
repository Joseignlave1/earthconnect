//! Native gRPC integration tests against the actual DuckDB serving snapshot.
mod common;
use earth_index_api::{
    AppState,
    proto::{self, earth_index_client::EarthIndexClient, value::Kind},
    router, value,
};
use prost::Message;
use std::collections::HashMap;
use tonic::{Code, Status, codec::CompressionEncoding, transport::Channel};

struct App {
    client: EarthIndexClient<Channel>,
    address: String,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for App {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn app(timeout: u64, max_size: usize) -> App {
    let database = common::database();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(
            listener,
            router(AppState {
                database,
                statement_timeout_ms: timeout,
                max_response_bytes: max_size,
                planetary: earth_index_api::planetary::Sampler::new().unwrap(),
            }),
        )
        .await
        .unwrap();
    });
    let client = EarthIndexClient::connect(address.clone())
        .await
        .unwrap()
        .max_decoding_message_size(16 << 20)
        .max_encoding_message_size(64 << 20)
        .send_compressed(CompressionEncoding::Gzip)
        .accept_compressed(CompressionEncoding::Gzip);
    App {
        client,
        address,
        task,
    }
}
async fn metadata(app: &App) -> proto::MetadataResponse {
    app.client
        .clone()
        .metadata(proto::MetadataRequest {})
        .await
        .unwrap()
        .into_inner()
}
async fn id(app: &App) -> String {
    metadata(app)
        .await
        .datasources
        .into_iter()
        .find(|d| d.id == "earth-ed5bc9904a5f")
        .unwrap()
        .id
}
async fn query(
    app: &App,
    id: &str,
    sql: &str,
    params: Vec<(&str, proto::Value)>,
) -> Result<proto::QueryResponse, Status> {
    app.client
        .clone()
        .query(proto::QueryRequest {
            query: sql.into(),
            params: params.into_iter().map(|(k, v)| (k.into(), v)).collect(),
            datasource_id: id.into(),
        })
        .await
        .map(|r| r.into_inner())
}
async fn ok(
    app: &App,
    id: &str,
    sql: &str,
    params: Vec<(&str, proto::Value)>,
) -> proto::QueryResponse {
    query(app, id, sql, params).await.unwrap()
}
fn int(v: i64) -> proto::Value {
    value::wrap(Kind::IntegerValue(v))
}
fn bytes(v: Vec<u8>) -> proto::Value {
    value::wrap(Kind::BytesValue(v))
}

#[tokio::test]
#[ignore = "requires prepared DuckDB snapshot"]
async fn discovers_real_coverage_and_schema() {
    let app = app(10000, 1 << 24).await;
    let data = metadata(&app).await;
    let largest = data
        .datasources
        .iter()
        .find(|d| d.id == "earth-ed5bc9904a5f")
        .unwrap();
    assert_eq!(largest.coverage.len(), 460);
    assert_eq!(largest.grids.len(), 460);
    assert!(largest.grids.iter().all(|g| g.width == 278
        && g.height == 278
        && (g.transform[0] - 3999.822013548645).abs() < 1e-6));
    assert!(
        largest
            .coverage
            .iter()
            .all(|c| c.srid == 4326 && !c.wkb.is_empty())
    );
    let cells = largest
        .relations
        .iter()
        .find(|r| r.name == "cells")
        .unwrap();
    let slope = cells
        .columns
        .iter()
        .find(|c| c.name == "slope_mean_deg")
        .unwrap();
    assert_eq!(slope.unit, "degree");
    assert_eq!(slope.provenance.len(), 460);
    assert!(
        app.client
            .clone()
            .health(proto::HealthRequest {})
            .await
            .unwrap()
            .into_inner()
            .ready
    );
}

#[tokio::test]
#[ignore = "requires prepared DuckDB snapshot"]
async fn exact_scalars_named_reuse_and_duplicate_columns() {
    let app = app(10000, 1 << 24).await;
    let id = id(&app).await;
    let r=ok(&app,&id,"SELECT :i AS v,:i+1 AS v,:f::float8,:b::boolean,:s::text,:n::integer,:date::date,:decimal::DECIMAL(38,18)",vec![("i",int(9007199254740993)),("f",value::wrap(Kind::NumberValue(1.25))),("b",value::wrap(Kind::BoolValue(false))),("s",value::string("'; DELETE FROM cells; -- ñ")),("n",value::null()),("date",value::string("2026-11-14")),("decimal",value::string("12345678901234567890.000000000000000001"))]).await;
    assert_eq!(r.columns[0].name, r.columns[1].name);
    let v = &r.rows[0].values;
    assert_eq!(v[0], int(9007199254740993));
    assert_eq!(v[1], int(9007199254740994));
    assert_eq!(v[5], value::null());
    assert_eq!(v[6], value::string("2026-11-14"));
    assert_eq!(
        v[7],
        value::string("12345678901234567890.000000000000000001")
    );
}

#[tokio::test]
#[ignore = "requires prepared DuckDB snapshot"]
async fn recursive_arrays_json_nulls_and_empty_results() {
    let app = app(10000, 1 << 24).await;
    let id = id(&app).await;
    let json = value::from_json(serde_json::json!({"items":[1,true,null,{"a":"b"}]}));
    let array = value::list(vec![
        value::list(vec![int(1), value::null()]),
        value::list(vec![int(3), int(4)]),
    ]);
    let r = ok(
        &app,
        &id,
        "SELECT :matrix::INTEGER[][],:json::JSON,:empty::VARCHAR[],:ja::INTEGER[]",
        vec![
            ("matrix", array.clone()),
            ("json", json.clone()),
            ("empty", value::list(vec![])),
            ("ja", value::list(vec![int(1), value::null()])),
        ],
    )
    .await;
    assert_eq!(r.rows[0].values[0], array);
    assert_eq!(r.rows[0].values[1], json);
    assert_eq!(r.rows[0].values[2], value::list(vec![]));
    assert_eq!(
        r.rows[0].values[3],
        value::list(vec![int(1), value::null()])
    );
    let r = ok(
        &app,
        &id,
        "SELECT site_key,x FROM cells WHERE false",
        vec![],
    )
    .await;
    assert_eq!(r.columns.len(), 2);
    assert!(r.rows.is_empty());
}

#[tokio::test]
#[ignore = "requires prepared DuckDB snapshot"]
async fn binary_geometry_and_large_payload_roundtrip() {
    let app = app(10000, 8 << 20).await;
    let id = id(&app).await;
    // Little-endian WKB polygon around Atacama.
    let mut polygon = vec![1];
    polygon.extend_from_slice(&3_u32.to_le_bytes());
    polygon.extend_from_slice(&1_u32.to_le_bytes());
    polygon.extend_from_slice(&5_u32.to_le_bytes());
    for (x, y) in [
        (-70_f64, -25_f64),
        (-67., -25.),
        (-67., -22.),
        (-70., -22.),
        (-70., -25.),
    ] {
        polygon.extend_from_slice(&x.to_le_bytes());
        polygon.extend_from_slice(&y.to_le_bytes());
    }
    let blob: Vec<u8> = (0..2_000_000).map(|i| (i % 256) as u8).collect();
    let r = ok(
        &app,
        &id,
        "SELECT :blob,ST_Area(ST_GeomFromWKB(:boundary)),ST_IsValid(ST_GeomFromWKB(:boundary))",
        vec![
            ("blob", bytes(blob.clone())),
            ("boundary", bytes(polygon.clone())),
        ],
    )
    .await;
    assert_eq!(r.rows[0].values[0], bytes(blob));
    assert_eq!(r.rows[0].values[1], value::wrap(Kind::NumberValue(9.)));
    let r=ok(&app,&id,"SELECT count(*) FROM cells WHERE site_key='h11v11' AND ST_Intersects(geom,ST_GeomFromWKB(:boundary))",vec![("boundary",bytes(polygon))]).await;
    assert!(matches!(r.rows[0].values[0].kind,Some(Kind::IntegerValue(n)) if n>0));
}

#[tokio::test]
#[ignore = "requires prepared DuckDB snapshot"]
async fn scoring_packing_and_snapshot_isolation() {
    let app = app(10000, 1 << 24).await;
    let id = id(&app).await;
    let r = ok(
        &app,
        &id,
        "SELECT count(*),count(DISTINCT region_id) FROM measurements",
        vec![],
    )
    .await;
    assert_eq!(r.rows[0].values, vec![int(35_550_640), int(460)]);
    let r = ok(
        &app,
        &id,
        "SELECT pack_u16(ARRAY[0,256,NULL,65535])",
        vec![],
    )
    .await;
    assert_eq!(
        r.rows[0].values[0],
        bytes(vec![0, 0, 0, 1, 255, 255, 255, 255])
    );
    let r=ok(&app,&id,"SELECT site_key,x,y,bare_fraction*exp(-slope_mean_deg/:scale) AS score FROM cells WHERE site_key=:site AND bare_fraction IS NOT NULL AND slope_mean_deg IS NOT NULL ORDER BY y,x LIMIT :limit",vec![("scale",int(8)),("site",value::string("h11v11")),("limit",int(10))]).await;
    assert_eq!(r.rows.len(), 10);
    assert!(
        r.rows.iter().all(
            |r| matches!(r.values[3].kind,Some(Kind::NumberValue(v))if(0.0..=1.0).contains(&v))
        )
    );
}

#[tokio::test]
#[ignore = "requires prepared DuckDB snapshot"]
async fn rejects_bad_requests_and_writes_without_poisoning_connections() {
    let app = app(10000, 1 << 24).await;
    let id = id(&app).await;
    for sql in [
        "SELECT :missing",
        "SELECT 1; SELECT 2",
        "COMMIT",
        "SET ROLE earth_index_owner",
        "WITH changed AS (DELETE FROM catalog.variables RETURNING *) SELECT * FROM changed",
        "SELECT does_not_exist",
    ] {
        let error = query(&app, &id, sql, vec![]).await.unwrap_err();
        assert!(
            matches!(error.code(), Code::InvalidArgument | Code::PermissionDenied),
            "{sql}: {error}"
        );
        assert!(!error.message().is_empty());
    }
    let error = query(&app, "unknown", "SELECT 1", vec![])
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::NotFound);
    let error = query(&app, "", "SELECT 1", vec![]).await.unwrap_err();
    assert_eq!(error.code(), Code::InvalidArgument);
    for sql in [
        "SELECT * FROM read_text('/etc/passwd')",
        "SELECT * FROM read_blob('/etc/passwd')",
        "SELECT * FROM read_csv('https://example.org/x.csv')",
    ] {
        assert!(query(&app, &id, sql, vec![]).await.is_err(), "{sql}");
    }
    let r = ok(
        &app,
        &id,
        "SELECT current_setting('enable_external_access'),current_setting('lock_configuration')",
        vec![],
    )
    .await;
    assert_eq!(
        r.rows[0].values,
        vec![
            value::wrap(Kind::BoolValue(false)),
            value::wrap(Kind::BoolValue(true))
        ]
    );
    ok(&app, &id, "SELECT 1", vec![]).await;
}

#[tokio::test]
#[ignore = "requires prepared DuckDB snapshot"]
async fn timeouts_and_response_limits_are_explicit() {
    let app = app(200, 128).await;
    let id = "earth-ed5bc9904a5f";
    let error = query(
        &app,
        id,
        "SELECT sum(sin(i::DOUBLE)) FROM range(10000000000) t(i)",
        vec![],
    )
    .await
    .unwrap_err();
    assert_eq!(error.code(), Code::DeadlineExceeded);
    if let Some(sqlstate) = error.metadata().get("sqlstate") {
        assert_eq!(sqlstate, "57014");
    }
    let error = query(&app, id, "SELECT repeat('x',1000)", vec![])
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::ResourceExhausted);
    ok(&app, id, "SELECT 1", vec![]).await;
}

#[test]
fn recursive_protocol_roundtrip_preserves_binary_and_int64() {
    let request = proto::QueryRequest {
        query: "SELECT :value".into(),
        datasource_id: "earth-test".into(),
        params: HashMap::from([(
            "value".into(),
            value::list(vec![int(i64::MAX), bytes(vec![0, 255]), value::null()]),
        )]),
    };
    assert_eq!(
        proto::QueryRequest::decode(request.encode_to_vec().as_slice()).unwrap(),
        request
    );
}

#[tokio::test]
#[ignore = "requires prepared DuckDB snapshot"]
async fn reflection_discovers_service_and_methods() {
    use tonic_reflection::pb::v1::{
        ServerReflectionRequest, server_reflection_client::ServerReflectionClient,
        server_reflection_request::MessageRequest, server_reflection_response::MessageResponse,
    };
    let app = app(10000, 1 << 24).await;
    let mut client = ServerReflectionClient::new(
        Channel::from_shared(app.address.clone())
            .unwrap()
            .connect()
            .await
            .unwrap(),
    );
    let request = ServerReflectionRequest {
        host: String::new(),
        message_request: Some(MessageRequest::ListServices(String::new())),
    };
    let mut stream = client
        .server_reflection_info(futures_util::stream::iter([request]))
        .await
        .unwrap()
        .into_inner();
    let response = stream.message().await.unwrap().unwrap();
    let Some(MessageResponse::ListServicesResponse(services)) = response.message_response else {
        panic!("Expected service list")
    };
    assert!(
        services
            .service
            .iter()
            .any(|s| s.name == "earth_index.v1.EarthIndex")
    );
    let request = ServerReflectionRequest {
        host: String::new(),
        message_request: Some(MessageRequest::FileContainingSymbol(
            "earth_index.v1.EarthIndex".into(),
        )),
    };
    let mut stream = client
        .server_reflection_info(futures_util::stream::iter([request]))
        .await
        .unwrap()
        .into_inner();
    let response = stream.message().await.unwrap().unwrap();
    let Some(MessageResponse::FileDescriptorResponse(files)) = response.message_response else {
        panic!("Expected descriptor")
    };
    let file = prost_types::FileDescriptorProto::decode(files.file_descriptor_proto[0].as_slice())
        .unwrap();
    let service = file
        .service
        .iter()
        .find(|s| s.name.as_deref() == Some("EarthIndex"))
        .unwrap();
    assert_eq!(
        service
            .method
            .iter()
            .map(|m| m.name.as_deref().unwrap())
            .collect::<Vec<_>>(),
        ["Query", "Metadata", "Health", "SamplePoint"]
    );
}

#[tokio::test]
#[ignore = "requires prepared DuckDB snapshot"]
async fn caller_deadline_cancels_database_work() {
    let app = app(0, 1 << 24).await;
    let id = id(&app).await;
    let mut request = tonic::Request::new(proto::QueryRequest {
        query: "SELECT sum(sin(i::DOUBLE)) FROM range(10000000000) t(i)".into(),
        params: HashMap::new(),
        datasource_id: id.clone(),
    });
    request.set_timeout(std::time::Duration::from_millis(30));
    let start = std::time::Instant::now();
    let error = app.client.clone().query(request).await.unwrap_err();
    assert!(
        matches!(error.code(), Code::DeadlineExceeded | Code::Cancelled),
        "{error}"
    );
    assert!(start.elapsed() < std::time::Duration::from_millis(500));
    ok(&app, &id, "SELECT 1", vec![]).await;
}

#[tokio::test]
#[ignore = "requires prepared DuckDB snapshot"]
async fn disabled_query_deadline_allows_work() {
    let app = app(0, 1 << 24).await;
    let id = id(&app).await;
    let response = ok(&app, &id, "SELECT sum(i) FROM range(10000000) t(i)", vec![]).await;
    assert_eq!(response.rows[0].values[0], int(49_999_995_000_000));
    ok(&app, &id, "SELECT 1", vec![]).await;
}
