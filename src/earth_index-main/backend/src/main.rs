use earth_index_api::{AppState, router};
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "earth_index_api=info,tower_http=info".into()),
        )
        .init();
    let database = earth_index_api::db::Database::open(
        std::env::var("DUCKDB_PATH").unwrap_or_else(|_| "db/runtime/earth.duckdb".into()),
        std::env::var("DUCKDB_SPATIAL_EXTENSION")
            .unwrap_or_else(|_| "db/runtime/spatial.duckdb_extension".into()),
        std::env::var("DUCKDB_THREADS")
            .unwrap_or_else(|_| "8".into())
            .parse()?,
        &std::env::var("DUCKDB_MEMORY_LIMIT").unwrap_or_else(|_| "8GB".into()),
    )?;
    let state = AppState {
        database,
        statement_timeout_ms: std::env::var("STATEMENT_TIMEOUT_MS")
            .unwrap_or_else(|_| "0".into())
            .parse()?,
        max_response_bytes: std::env::var("MAX_RESPONSE_BYTES")
            .unwrap_or_else(|_| "134217728".into())
            .parse()?,
        planetary: earth_index_api::planetary::Sampler::new()?,
    };
    let address = std::env::var("API_LISTEN").unwrap_or_else(|_| "0.0.0.0:8080".into());
    let listener = tokio::net::TcpListener::bind(&address).await?;
    tracing::info!(%address,"Earth Index gRPC + gRPC-Web listening; explorer at /docs");
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
