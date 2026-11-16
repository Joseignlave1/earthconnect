mod common;
use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use earth_index_api::{AppState, router};
use http_body_util::BodyExt;
use tower::ServiceExt;

#[tokio::test]
#[ignore = "requires npm --prefix demo run build"]
async fn serves_demo_bundle_and_overview_tiles() {
    let app = router(AppState {
        database: common::database(),
        statement_timeout_ms: 10000,
        max_response_bytes: 1 << 20,
        planetary: earth_index_api::planetary::Sampler::new().unwrap(),
    });
    let response = app
        .clone()
        .oneshot(Request::builder().uri("/demo").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let html = String::from_utf8(
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .to_vec(),
    )
    .unwrap();
    // Svelte mounts the map panels into this root after loading the bundle.
    assert!(html.contains("id=\"app\""));
    let asset = html
        .split("src=\"")
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap();
    assert!(asset.starts_with("/demo/assets/"));
    let response = app
        .clone()
        .oneshot(Request::builder().uri(asset).body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert!(
        response.headers()["content-type"]
            .to_str()
            .unwrap()
            .contains("javascript")
    );
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/demo/overview/moon-0.jpg")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "image/jpeg");
    let response = app
        .oneshot(
            Request::builder()
                .uri("/demo/missing.js")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}
