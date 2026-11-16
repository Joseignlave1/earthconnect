//! Schema downloads and the standard reflection-driven grpcui explorer.
use crate::{AppState, DESCRIPTOR};
use axum::{Router, response::Redirect, routing::get};
use std::sync::Arc;

pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/", get(explorer))
        .route("/docs", get(explorer))
        .route(
            "/proto/earth_index.proto",
            get(|| async {
                (
                    [("content-type", "text/plain; charset=utf-8")],
                    include_str!("../proto/earth_index.proto"),
                )
            }),
        )
        .route(
            "/proto/google/protobuf/struct.proto",
            get(|| async {
                (
                    [("content-type", "text/plain; charset=utf-8")],
                    include_str!("../proto/google/protobuf/struct.proto"),
                )
            }),
        )
        .route(
            "/proto/descriptor.bin",
            get(|| async { ([("content-type", "application/octet-stream")], DESCRIPTOR) }),
        )
}
async fn explorer() -> Redirect {
    Redirect::temporary(
        &std::env::var("GRPCUI_URL").unwrap_or_else(|_| "http://localhost:8081".into()),
    )
}
