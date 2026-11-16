use crate::{AppState, error::ApiError, proto};
use duckdb::Connection;

pub async fn load(state: &AppState) -> Result<proto::MetadataResponse, ApiError> {
    Ok(state.database.metadata.clone())
}

// The frozen metadata retains the scientific source precision/provenance.
// Serving storage types and SQL helpers describe the new engine accurately.
pub fn describe_duckdb(
    metadata: &mut proto::MetadataResponse,
    conn: &Connection,
) -> duckdb::Result<()> {
    for datasource in &mut metadata.datasources {
        datasource.relations.retain(|r| r.name == "cells");
        let relation = &mut datasource.relations[0];
        relation.description = "Physical values in DOUBLE columns; categorical codes and addresses are integers. x/y are 1-based. geom is WGS84 and NULL for padding. Rows have no implicit order.".into();
        let mut stmt = conn.prepare("DESCRIBE cells")?;
        let columns = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?
            .collect::<duckdb::Result<Vec<_>>>()?;
        relation
            .columns
            .retain(|c| columns.iter().any(|(name, _)| name == &c.name));
        for (name, ty) in columns {
            if let Some(column) = relation.columns.iter_mut().find(|c| c.name == name) {
                column.pg_type = ty;
            } else {
                relation.columns.push(proto::ColumnMetadata {
                    name,
                    pg_type: ty,
                    nullable: true,
                    ..Default::default()
                });
            }
        }
        for name in ["measurements", "regions"] {
            let mut stmt = conn.prepare(&format!("DESCRIBE {name}"))?;
            let columns = stmt
                .query_map([], |r| {
                    Ok(proto::ColumnMetadata {
                        name: r.get(0)?,
                        pg_type: r.get(1)?,
                        nullable: true,
                        ..Default::default()
                    })
                })?
                .collect::<duckdb::Result<Vec<_>>>()?;
            datasource.relations.push(proto::Relation {name:name.into(),description:if name=="measurements" {"Fast physical column store; region_id and zero-based row-major pixel identify a cell. Same measurement fields as cells, without computed geometry or a region join."} else {"One row per native region with shape, affine transform, CRS and WGS84 loaded-coverage geometry."}.into(),columns});
        }
        datasource.functions = vec![
            proto::Function { signature: "pack_u16_at(STRUCT(index UINTEGER, code INTEGER)[], UINTEGER) → BLOB".into(), description: "Scatter zero-based cell addresses into a fixed-length little-endian uint16 matrix without sorting. Missing addresses/NULL codes become 65535. Duplicate/out-of-range addresses or codes are errors. Length is capped at 5 million cells.".into() },
            proto::Function { signature: "pack_u16(INTEGER[]) → BLOB".into(), description: "Little-endian uint16 packing, preserving list order. NULL elements become 65535; accepts 0..65535. Quantize explicitly in SQL with round_even(65534 * score, 0)::INTEGER.".into() },
            proto::Function { signature: "DuckDB SQL + Spatial".into(), description: "Native DuckDB expressions, lists, structs and Spatial functions; geometry results are WKB bytes. Bind binary polygons with ST_GeomFromWKB(:boundary). The pg_type protobuf field carries DuckDB types for wire compatibility.".into() },
        ];
    }
    Ok(())
}
