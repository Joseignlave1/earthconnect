use crate::AppState;
use axum::Router;
use std::{path::PathBuf, sync::Arc};
use tower_http::services::ServeDir;

pub fn routes() -> Router<Arc<AppState>> {
    let directory = std::env::var_os("DEMO_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../demo/dist"));
    Router::new().nest_service("/demo", ServeDir::new(directory))
}
