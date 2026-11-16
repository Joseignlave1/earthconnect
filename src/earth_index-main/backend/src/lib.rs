pub mod db;
pub mod demo;
pub mod docs;
pub mod error;
pub mod metadata;
mod packing;
pub mod params;
pub mod planetary;
pub mod rpc;
pub mod sql;
pub mod value;

pub mod proto {
    tonic::include_proto!("earth_index.v1");
}

use axum::{
    Router,
    extract::State,
    http::{Method, StatusCode},
    routing::get,
};
use std::sync::Arc;
use tonic::{codec::CompressionEncoding, service::Routes};
use tower::Layer;
use tower_http::{
    compression::{CompressionLayer, CompressionLevel},
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};

pub const DESCRIPTOR: &[u8] = tonic::include_file_descriptor_set!("descriptor");
pub const MAX_REQUEST_BYTES: usize = 64 * 1024 * 1024;

#[derive(Clone)]
pub struct AppState {
    pub database: Arc<db::Database>,
    /// Query execution budget; zero disables the server-imposed deadline.
    pub statement_timeout_ms: u64,
    pub max_response_bytes: usize,
    pub planetary: planetary::Sampler,
}

pub fn router(state: AppState) -> Router {
    let state = Arc::new(state);
    let api =
        proto::earth_index_server::EarthIndexServer::new(rpc::EarthIndexService(state.clone()))
            .max_decoding_message_size(MAX_REQUEST_BYTES)
            .max_encoding_message_size(state.max_response_bytes)
            .accept_compressed(CompressionEncoding::Gzip)
            .send_compressed(CompressionEncoding::Gzip);
    let reflection = || {
        tonic_reflection::server::Builder::configure()
            .register_encoded_file_descriptor_set(DESCRIPTOR)
    };
    let rpc_routes = Routes::new(tonic_web::GrpcWebLayer::new().layer(api))
        .add_service(
            reflection()
                .build_v1()
                .expect("compiled reflection descriptor"),
        )
        .add_service(
            reflection()
                .build_v1alpha()
                .expect("compiled reflection descriptor"),
        )
        .into_axum_router()
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods([Method::POST, Method::OPTIONS])
                .allow_headers(Any)
                .expose_headers(Any),
        );
    Router::new()
        .route("/healthz", get(healthz))
        .merge(docs::routes())
        .merge(demo::routes())
        .with_state(state)
        .merge(rpc_routes)
        // HTTP encoding is transparent to browser gRPC-Web clients. The default
        // predicate excludes native gRPC, which negotiates its own compression.
        .layer(CompressionLayer::new().quality(CompressionLevel::Precise(1)))
        .layer(TraceLayer::new_for_grpc())
}

pub async fn ready(state: &AppState) -> bool {
    !state.database.metadata.datasources.is_empty()
}

async fn healthz(State(state): State<Arc<AppState>>) -> (StatusCode, &'static str) {
    if ready(&state).await {
        (StatusCode::OK, "ready\n")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "not ready\n")
    }
}
