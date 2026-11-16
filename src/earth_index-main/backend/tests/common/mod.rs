use earth_index_api::db::Database;
use std::sync::{Arc, OnceLock};
pub fn database() -> Arc<Database> {
    static DB: OnceLock<Arc<Database>> = OnceLock::new();
    DB.get_or_init(|| {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../db/runtime");
        Database::open(
            root.join("earth.duckdb"),
            root.join("spatial.duckdb_extension"),
            8,
            "8GB",
        )
        .expect("Run uv run db/prepare_duckdb.py first")
    })
    .clone()
}
