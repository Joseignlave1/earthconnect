use super::*;
use crate::{
    AppState,
    proto::{earth_index_client::EarthIndexClient, value::Kind as ValueKind},
    router, value,
};
use axum::{
    Router,
    body::Body,
    extract::State,
    http::{Request, StatusCode, Uri},
    response::IntoResponse,
    routing::get,
};
use http_body_util::BodyExt;
use prost::Message;
use std::sync::atomic::{AtomicUsize, Ordering};
use tower::ServiceExt;

struct Server {
    address: String,
    task: tokio::task::JoinHandle<()>,
}
impl Drop for Server {
    fn drop(&mut self) {
        self.task.abort();
    }
}
async fn server(app: Router) -> Server {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    Server { address, task }
}
fn app(sampler: Sampler) -> Router {
    router(AppState {
        database: crate::db::Database::test_database(),
        statement_timeout_ms: 10000,
        max_response_bytes: 1 << 20,
        planetary: sampler,
    })
}
fn request(body: PlanetaryBody) -> SamplePointRequest {
    SamplePointRequest {
        body: body.into(),
        longitude: 23.47,
        latitude: 0.67,
    }
}
fn fixture_source(kind: Kind, id: &str, url: &str) -> Source {
    let mut source = sources::registry()
        .into_iter()
        .find(|s| matches!(s.kind, Kind::MarsWms))
        .unwrap();
    source.url = url.into();
    source.variable_id = id.into();
    source.source_id = id.into();
    source.kind = kind;
    source.nodata = vec![-32768.];
    source.valid_range = None;
    source.layer_id = Some(0);
    source
}
fn client(timeout: Duration) -> Client {
    Client::builder()
        .timeout(timeout)
        .no_proxy()
        .build()
        .unwrap()
}
fn number(m: &PointMeasurement) -> f64 {
    match m.value.as_ref().unwrap().kind.as_ref().unwrap() {
        ValueKind::NumberValue(v) => *v,
        other => panic!("unexpected {other:?}"),
    }
}
fn vicar(format: &str, real: &str, integer: &str, pixel: &[u8]) -> Vec<u8> {
    let header = format!(
        "LBLSIZE=256 HOST='PC' INTFMT='{integer}' REALFMT='{real}' FORMAT='{format}' ORG='BSQ' NL=1 NS=1 NB=1 NBB=0 NLB=0 RECSIZE={} TASK='test task'",
        pixel.len()
    );
    let mut body = header.into_bytes();
    body.resize(256, b' ');
    body.extend_from_slice(pixel);
    body
}

#[test]
fn audited_registry_is_unique_and_excludes_bad_and_rendered_layers() {
    let sources = sources::registry();
    assert_eq!(sources.len(), 79);
    let mut ids = std::collections::HashSet::new();
    for source in &sources {
        assert!(ids.insert((&source.variable_id, &source.source_id)));
        assert!(
            source.url.starts_with("https://trek.nasa.gov/")
                || source.url == "https://ms-mars.mars.asu.edu/alb32"
        );
        assert!(!source.source_id.contains("temperature_range_celsius"));
        if source
            .quality_flags
            .iter()
            .any(|flag| flag == "semantic_review_required")
        {
            assert_eq!(source.unit, "raw");
        }
        if matches!(source.kind, Kind::MarsWms) {
            assert!(source.ppd.unwrap() > 0.);
        }
    }
}

#[test]
fn planetary_footprints_and_physical_masks() {
    let sources = sources::registry();
    let source = |id| sources.iter().find(|s| s.source_id == id).unwrap();
    assert!(source("AVGVISIB_65N_240M_201608_NP").covers(0., 85.));
    assert!(!source("AVGVISIB_65N_240M_201608_NP").covers(0., -85.));
    assert!(source("AVGVISIB_65S_240M_201608_SP").covers(0., -85.));
    assert!(source("LRO_NAC_RockAbundance_100mpp_SiteB").covers(162.6293, -57.22227));
    assert!(!source("LRO_NAC_RockAbundance_100mpp_SiteB").covers(0., 0.));
    assert!(!source("Lunar_Kaguya_MIMap_MineralDeconv_OlivinePercent_50N50S").covers(0., 75.));
    assert!(!source("dgdr_std_cf_clc_cyl_128_jp2").valid(-25.77));
    assert!(source("LRO_LOLA_DEM_Global_256ppd_v06").valid(0.));
    assert!(!source("TES_dust_index_16ppd_numeric").valid(0.85000002));
}

#[test]
fn vicar_real_integer_double_and_failure_fixtures() {
    for (format, real, integer, pixel, want) in [
        (
            "REAL",
            "RIEEE",
            "LOW",
            0.125f32.to_le_bytes().to_vec(),
            0.125,
        ),
        (
            "HALF",
            "IEEE",
            "HIGH",
            (-123i16).to_be_bytes().to_vec(),
            -123.,
        ),
        ("FULL", "RIEEE", "LOW", 209i32.to_le_bytes().to_vec(), 209.),
        (
            "DOUB",
            "IEEE",
            "HIGH",
            9.875f64.to_be_bytes().to_vec(),
            9.875,
        ),
    ] {
        assert_eq!(
            mars::decode_pixel(&vicar(format, real, integer, &pixel)).unwrap(),
            want
        );
    }
    assert!(mars::decode_pixel(b"<ServiceException>InvalidFormat</ServiceException>").is_err());
    let payload = vicar("REAL", "RIEEE", "LOW", &1f32.to_le_bytes());
    assert!(mars::decode_pixel(&payload[..payload.len() - 1]).is_err());
    assert!(mars::decode_pixel(&vicar("REAL", "VAX", "LOW", &1f32.to_le_bytes())).is_err());
}

#[test]
fn feature_empty_is_missing_and_overflow_is_explicit() {
    let source = fixture_source(Kind::TrekFeature, "mars.geology", "http://localhost");
    let mut m = source.template();
    trek::parse(&source, br#"{"features":[]}"#, &mut m).unwrap();
    assert_eq!(m.status(), MeasurementStatus::NoData);
    assert!(m.value.is_none());
    trek::parse(&source, br#"{"features":[{"attributes":{"unit":"HNhu","missing":null}}],"exceededTransferLimit":true}"#, &mut m).unwrap();
    assert_eq!(m.status(), MeasurementStatus::Available);
    assert_eq!(
        value::to_json(m.value.as_ref().unwrap()).unwrap(),
        serde_json::json!([{"unit":"HNhu","missing":null}])
    );
    assert!(
        m.quality_flags
            .iter()
            .any(|flag| flag == "truncated_feature_matches")
    );
    assert!(
        trek::parse(
            &source,
            br#"{"error":{"code":400}}"#,
            &mut source.template()
        )
        .is_err()
    );
}

#[test]
fn derives_range_without_a_celsius_offset_or_inventing_missing_inputs() {
    let make = |id: &str, n| PointMeasurement {
        variable_id: id.into(),
        status: MeasurementStatus::Available.into(),
        value: Some(value::wrap(ValueKind::NumberValue(n))),
        pixel_spacing: 0.125,
        pixel_spacing_unit: "degree".into(),
        ..Default::default()
    };
    let mut m = vec![
        make("mars.modeled_surface_temperature_min", -120.),
        make("mars.modeled_surface_temperature_max", 20.),
    ];
    derive_temperature_range(PlanetaryBody::Mars, &mut m);
    assert_eq!(number(m.last().unwrap()), 140.);
    m.truncate(2);
    m[0].status = MeasurementStatus::NoData.into();
    m[0].value = None;
    derive_temperature_range(PlanetaryBody::Mars, &mut m);
    assert_eq!(m.last().unwrap().status(), MeasurementStatus::NoData);
}

#[tokio::test]
async fn native_and_web_rpc_return_partial_data_and_reuse_the_same_cache() {
    let hits = Arc::new(AtomicUsize::new(0));
    async fn raster(State(hits): State<Arc<AtomicUsize>>) -> impl IntoResponse {
        hits.fetch_add(1, Ordering::SeqCst);
        axum::Json(serde_json::json!({"samples":[{"value":"0","resolution":0.125}]}))
    }
    let upstream = server(
        Router::new()
            .route("/good/getSamples", get(raster))
            .route(
                "/bad/getSamples",
                get(|| async { axum::Json(serde_json::json!({"error":{"code":503}})) }),
            )
            .with_state(hits.clone()),
    )
    .await;
    let sampler = Sampler::with_sources(
        client(Duration::from_secs(1)),
        vec![
            fixture_source(
                Kind::TrekRaster,
                "mars.zero",
                &format!("{}/good", upstream.address),
            ),
            fixture_source(
                Kind::TrekRaster,
                "mars.bad",
                &format!("{}/bad", upstream.address),
            ),
        ],
    );
    let api = app(sampler.clone());
    let native = server(api.clone()).await;
    let mut rpc = EarthIndexClient::connect(native.address.clone())
        .await
        .unwrap();
    let response = rpc
        .sample_point(request(PlanetaryBody::Mars))
        .await
        .unwrap()
        .into_inner();
    assert!(response.partial);
    assert!(!response.cached);
    assert_eq!(number(&response.measurements[0]), 0.);
    assert_eq!(
        response.measurements[1].status(),
        MeasurementStatus::SourceError
    );
    let message = request(PlanetaryBody::Mars).encode_to_vec();
    let mut frame = vec![0];
    frame.extend_from_slice(&(message.len() as u32).to_be_bytes());
    frame.extend_from_slice(&message);
    let response = api
        .oneshot(
            Request::post("/earth_index.v1.EarthIndex/SamplePoint")
                .header("content-type", "application/grpc-web+proto")
                .header("x-grpc-web", "1")
                .header("origin", "https://example.com")
                .body(Body::from(frame))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["access-control-allow-origin"], "*");
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(bytes[0], 0);
    let n = u32::from_be_bytes(bytes[1..5].try_into().unwrap()) as usize;
    let response = SamplePointResponse::decode(&bytes[5..5 + n]).unwrap();
    assert!(response.cached);
    assert!(response.partial);
    assert_eq!(hits.load(Ordering::SeqCst), 1);
    let cached = sampler
        .sample(request(PlanetaryBody::Mars), Duration::from_secs(1))
        .await
        .unwrap();
    assert!(cached.cached);
    assert_eq!(cached.retrieved_at, response.retrieved_at);
}

#[tokio::test]
async fn wms_requests_are_tiny_numeric_and_do_not_encode_the_format_slash() {
    let upstream = server(Router::new().route(
        "/wms",
        get(|uri: Uri| async move {
            let query = uri.query().unwrap();
            assert!(query.contains("FORMAT=image/vicar"));
            assert!(query.contains("WIDTH=1&HEIGHT=1"));
            assert!(query.contains("SRS=JMARS:4"));
            vicar("REAL", "RIEEE", "LOW", &0.125f32.to_le_bytes())
        }),
    ))
    .await;
    let sampler = Sampler::with_sources(
        client(Duration::from_secs(1)),
        vec![fixture_source(
            Kind::MarsWms,
            "mars.albedo",
            &format!("{}/wms", upstream.address),
        )],
    );
    let response = sampler
        .sample(request(PlanetaryBody::Mars), Duration::from_secs(1))
        .await
        .unwrap();
    assert!(!response.partial);
    assert_eq!(number(&response.measurements[0]), 0.125);
}

#[tokio::test]
async fn overall_budget_returns_fast_successes_and_cancels_slow_sources() {
    let upstream = server(
        Router::new()
            .route(
                "/fast/getSamples",
                get(|| async { axum::Json(serde_json::json!({"samples":[{"value":"42"}]})) }),
            )
            .route(
                "/slow/getSamples",
                get(|| async {
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    "{}"
                }),
            ),
    )
    .await;
    let sampler = Sampler::with_sources(
        client(Duration::from_secs(5)),
        vec![
            fixture_source(
                Kind::TrekRaster,
                "mars.fast",
                &format!("{}/fast", upstream.address),
            ),
            fixture_source(
                Kind::TrekRaster,
                "mars.slow",
                &format!("{}/slow", upstream.address),
            ),
        ],
    );
    let start = Instant::now();
    let response = sampler
        .sample(request(PlanetaryBody::Mars), Duration::from_millis(150))
        .await
        .unwrap();
    assert!(start.elapsed() < Duration::from_secs(1));
    assert!(response.partial);
    assert_eq!(number(&response.measurements[0]), 42.);
    assert_eq!(
        response.measurements[1].status(),
        MeasurementStatus::SourceTimeout
    );
    assert_eq!(sampler.0.upstream.available_permits(), 8);
}

#[tokio::test]
async fn same_point_requests_coalesce_and_invalid_coordinates_make_no_requests() {
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = hits.clone();
    let upstream = server(Router::new().route(
        "/getSamples",
        get(move || {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(20)).await;
                axum::Json(serde_json::json!({"samples":[{"value":"12"}]}))
            }
        }),
    ))
    .await;
    let sampler = Sampler::with_sources(
        client(Duration::from_secs(1)),
        vec![fixture_source(
            Kind::TrekRaster,
            "mars.point",
            &upstream.address,
        )],
    );
    for (body, lon, lat) in [
        (0, 0., 0.),
        (99, 0., 0.),
        (2, f64::NAN, 0.),
        (2, 181., 0.),
        (1, 0., 91.),
    ] {
        let error = sampler
            .sample(
                SamplePointRequest {
                    body,
                    longitude: lon,
                    latitude: lat,
                },
                Duration::from_secs(1),
            )
            .await
            .unwrap_err();
        assert_eq!(error.code(), tonic::Code::InvalidArgument);
    }
    assert_eq!(hits.load(Ordering::SeqCst), 0);
    let (a, b) = tokio::join!(
        sampler.sample(request(PlanetaryBody::Mars), Duration::from_secs(1)),
        sampler.sample(request(PlanetaryBody::Mars), Duration::from_secs(1))
    );
    assert_ne!(a.unwrap().cached, b.unwrap().cached);
    assert_eq!(hits.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn oversized_and_http_error_sources_are_partial_without_values() {
    let upstream = server(
        Router::new()
            .route(
                "/large/getSamples",
                get(|| async { vec![b'x'; MAX_UPSTREAM_BYTES + 1] }),
            )
            .route(
                "/rate/getSamples",
                get(|| async { StatusCode::TOO_MANY_REQUESTS }),
            ),
    )
    .await;
    let sampler = Sampler::with_sources(
        client(Duration::from_secs(1)),
        vec![
            fixture_source(
                Kind::TrekRaster,
                "mars.large",
                &format!("{}/large", upstream.address),
            ),
            fixture_source(
                Kind::TrekRaster,
                "mars.rate",
                &format!("{}/rate", upstream.address),
            ),
        ],
    );
    let response = sampler
        .sample(request(PlanetaryBody::Mars), Duration::from_secs(1))
        .await
        .unwrap();
    assert!(response.partial);
    assert!(
        response
            .measurements
            .iter()
            .all(|m| m.status() == MeasurementStatus::SourceError && m.value.is_none())
    );
    assert!(response.measurements[1].reason.contains("429"));
}

#[tokio::test]
async fn complete_vicar_pixel_does_not_require_a_transport_eof() {
    let upstream = server(Router::new().route(
        "/wms",
        get(|| async {
            let data = stream::once(async {
                Ok::<_, std::io::Error>(vicar("REAL", "RIEEE", "LOW", &0.25f32.to_le_bytes()))
            });
            // Reproduce a complete self-framed raster whose HTTP/TLS terminator is
            // absent. The sampler must not wait for or trust an incomplete pixel.
            axum::response::Response::new(Body::from_stream(data.chain(stream::pending())))
        }),
    ))
    .await;
    let sampler = Sampler::with_sources(
        client(Duration::from_secs(3)),
        vec![fixture_source(
            Kind::MarsWms,
            "mars.albedo",
            &format!("{}/wms", upstream.address),
        )],
    );
    let start = Instant::now();
    let response = sampler
        .sample(request(PlanetaryBody::Mars), Duration::from_secs(2))
        .await
        .unwrap();
    assert!(start.elapsed() < Duration::from_secs(1));
    assert!(!response.partial);
    assert_eq!(number(&response.measurements[0]), 0.25);
}

#[test]
fn local_terrain_stencil_preserves_center_and_uses_planetary_distances() {
    for body in ["moon", "mars"] {
        let source = sources::registry()
            .into_iter()
            .find(|s| s.body == body && terrain::enabled(s))
            .unwrap();
        let latitude = 30.0_f64;
        let radius = if body == "moon" {
            1_737_400.
        } else {
            3_396_190.
        };
        let dx = radius * source.pixel_spacing.to_radians() * latitude.to_radians().cos();
        let samples: Vec<_> = (0..9)
            .rev()
            .map(|i| serde_json::json!({"locationId":i,"value":(i%3-1) as f64*dx*0.1}))
            .collect();
        let bytes = serde_json::to_vec(&serde_json::json!({"samples":samples})).unwrap();
        let mut center = source.template();
        let derived = terrain::derive(&source, &bytes, latitude, &mut center);
        assert_eq!(number(&center), 0.);
        assert!((number(&derived[0]) - 0.1_f64.atan().to_degrees()).abs() < 1e-10);
        assert!((number(&derived[1]) - dx * 0.2).abs() < 1e-8);
        let points = terrain::points(&source, 179.99999, latitude);
        assert!(points.iter().all(|p| (-180. ..180.).contains(&p[0])));
        let missing = terrain::derive(
            &source,
            br#"{"samples":[{"locationId":4,"value":"0"}]}"#,
            latitude,
            &mut center,
        );
        assert_eq!(number(&center), 0.);
        assert!(
            missing
                .iter()
                .all(|m| m.status() == MeasurementStatus::NoData)
        );
        assert!(!terrain::has_stencil(&source, 90.));
    }
}

#[test]
fn kilometre_windows_preserve_scale_and_ignore_vertical_datums() {
    for body in ["moon", "mars"] {
        let source = sources::registry()
            .into_iter()
            .find(|s| s.body == body && terrain::enabled(s))
            .unwrap();
        for offset in [0., 5000.] {
            let samples: Vec<_> = (0..441).rev().map(|i| serde_json::json!({"locationId":i,"value":offset + (i % 21) as f64 * 10. + (i / 21) as f64 * 20.})).collect();
            let result = terrain::derive_window(&source, &serde_json::json!({"samples":samples}));
            assert!((number(&result[0]) - 0.01_f64.hypot(0.02).atan().to_degrees()).abs() < 1e-10);
            assert_eq!(number(&result[1]), 120.);
            assert_eq!(number(&result[2]), 0.);
            assert_eq!(number(&result[3]), 600.);
            assert_eq!(number(&result[4]), 0.);
        }
        for latitude in [-90., -89.99, 0., 89.99, 90.] {
            let points = terrain::window_points(&source, 179.9999, latitude);
            assert_eq!(points.len(), 441);
            assert!(points.iter().all(|[lon, lat]| lon.is_finite()
                && lat.is_finite()
                && (-180. ..180.).contains(lon)
                && (-90. ..=90.).contains(lat)));
            let a = points[220];
            let b = points[221];
            let h = ((b[1] - a[1]).to_radians() / 2.).sin().powi(2)
                + a[1].to_radians().cos()
                    * b[1].to_radians().cos()
                    * ((b[0] - a[0]).to_radians() / 2.).sin().powi(2);
            let radius = if body == "moon" {
                1_737_400.
            } else {
                3_396_190.
            };
            assert!((2. * radius * h.sqrt().asin() - 1000.).abs() < 0.1);
        }
    }
}

#[test]
fn kilometre_windows_require_a_center_and_ninety_percent_support() {
    let source = sources::registry()
        .into_iter()
        .find(terrain::enabled)
        .unwrap();
    let make = |omit: usize, center: bool| serde_json::json!({"samples":(0..441).filter(|i| *i >= omit && (!center || *i != 220)).map(|i| serde_json::json!({"locationId":i,"value":0})).collect::<Vec<_>>()});
    let enough = terrain::derive_window(&source, &make(44, false));
    assert_eq!(number(&enough[3]), 0.);
    assert!(
        enough[3]
            .quality_flags
            .contains(&"incomplete_terrain_window".into())
    );
    assert_eq!(
        terrain::derive_window(&source, &make(45, false))[3].status(),
        MeasurementStatus::NoData
    );
    assert!(
        terrain::derive_window(&source, &make(0, true))
            .iter()
            .all(|m| m.status() == MeasurementStatus::NoData)
    );
}

#[tokio::test]
async fn kilometre_sampling_batches_small_gets_and_reassembles_local_ids() {
    let hits = Arc::new(AtomicUsize::new(0));
    let counter = hits.clone();
    let upstream = server(Router::new().route("/getSamples",get(move |uri: Uri| {
        let counter=counter.clone();
        async move {
            counter.fetch_add(1,Ordering::SeqCst);
            assert!(uri.to_string().len()<4000);
            let url=reqwest::Url::parse(&format!("http://local{uri}")).unwrap();
            let geometry=url.query_pairs().find(|(k,_)|k=="geometry").unwrap().1.into_owned();
            let geometry:serde_json::Value=serde_json::from_str(&geometry).unwrap();
            let points=geometry["points"].as_array().unwrap();
            assert!(points.len()<=48);
            let samples:Vec<_>=points.iter().enumerate().rev().map(|(i,_)|serde_json::json!({"locationId":i,"value":"100","resolution":0.003})).collect();
            axum::Json(serde_json::json!({"samples":samples}))
        }
    }))).await;
    let mut source = sources::registry()
        .into_iter()
        .find(|s| s.body == "mars" && terrain::enabled(s))
        .unwrap();
    source.url = upstream.address.clone();
    let sampler = Sampler::with_sources(client(Duration::from_secs(1)), vec![source]);
    let response = sampler
        .sample(request(PlanetaryBody::Mars), Duration::from_secs(3))
        .await
        .unwrap();
    assert!(!response.partial);
    assert_eq!(hits.load(Ordering::SeqCst), 11);
    assert_eq!(response.measurements.len(), 8); // source + native two + window five
    let relief = response
        .measurements
        .iter()
        .find(|m| m.variable_id == "mars.relief_21km")
        .unwrap();
    assert_eq!(number(relief), 0.);
    assert_eq!(sampler.0.upstream.available_permits(), 8);
}

#[tokio::test]
async fn stalled_window_preserves_successful_center_before_total_deadline() {
    let upstream=server(Router::new().route("/getSamples",get(|uri:Uri|async move {
        let url=reqwest::Url::parse(&format!("http://local{uri}")).unwrap();
        let geometry=url.query_pairs().find(|(k,_)|k=="geometry").unwrap().1.into_owned();
        let geometry:serde_json::Value=serde_json::from_str(&geometry).unwrap();
        let count=geometry["points"].as_array().unwrap().len();
        if count>9 { tokio::time::sleep(Duration::from_secs(10)).await; }
        axum::Json(serde_json::json!({"samples":(0..count).map(|i|serde_json::json!({"locationId":i,"value":100})).collect::<Vec<_>>()}))
    }))).await;
    let mut source = sources::registry()
        .into_iter()
        .find(|s| s.body == "mars" && terrain::enabled(s))
        .unwrap();
    source.url = upstream.address.clone();
    let sampler = Sampler::with_sources(client(Duration::from_secs(5)), vec![source]);
    let response = sampler
        .sample(request(PlanetaryBody::Mars), Duration::from_millis(500))
        .await
        .unwrap();
    assert!(response.partial);
    assert_eq!(number(&response.measurements[0]), 100.);
    assert_eq!(
        response.measurements[0].status(),
        MeasurementStatus::Available
    );
    assert_eq!(
        response
            .measurements
            .iter()
            .filter(|m| m
                .quality_flags
                .contains(&"kilometre_terrain_estimate".into())
                && m.status() == MeasurementStatus::NoData)
            .count(),
        5
    );
    assert_eq!(sampler.0.upstream.available_permits(), 8);
}
