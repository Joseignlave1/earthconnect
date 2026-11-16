//! cargo run --manifest-path backend/Cargo.toml --example sample_point -- moon 0 0
use earth_index_api::{
    proto::{PlanetaryBody, SamplePointRequest, earth_index_client::EarthIndexClient},
    value,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: sample_point moon|mars LONGITUDE LATITUDE (API_URL defaults to http://localhost:8080)".into());
    }
    let body = match args[1].as_str() {
        "moon" => PlanetaryBody::Moon,
        "mars" => PlanetaryBody::Mars,
        _ => return Err("body must be moon or mars".into()),
    };
    let url = std::env::var("API_URL").unwrap_or_else(|_| "http://localhost:8080".into());
    let mut client = EarthIndexClient::connect(url).await?;
    let response = client
        .sample_point(SamplePointRequest {
            body: body.into(),
            longitude: args[2].parse()?,
            latitude: args[3].parse()?,
        })
        .await?
        .into_inner();
    let measurements: Vec<_> = response.measurements.iter().map(|m| serde_json::json!({
        "variable_id": m.variable_id, "source_id": m.source_id, "status": m.status().as_str_name(),
        "value": m.value.as_ref().map(value::to_json).transpose().unwrap(), "unit": m.unit,
        "pixel_spacing": m.pixel_spacing, "pixel_spacing_unit": m.pixel_spacing_unit,
        "quality_flags":m.quality_flags, "notes": m.notes,"reason":m.reason,"source_url":m.source_url
    })).collect();
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"body":args[1], "longitude":response.longitude,"latitude":response.latitude,
        "partial":response.partial,"cached":response.cached,"retrieved_at":response.retrieved_at,"measurements":measurements})
        )?
    );
    Ok(())
}
