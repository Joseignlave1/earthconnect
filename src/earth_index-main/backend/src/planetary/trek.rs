use super::sources::{Kind, Source};
use crate::{
    proto::{self, MeasurementStatus, PointMeasurement},
    value,
};
use serde_json::{Value, json};

pub(super) fn multipoint_url(source: &Source, points: &[[f64; 2]]) -> reqwest::Url {
    let sr = if source.body == "moon" {
        104903
    } else {
        104905
    };
    let mut url =
        reqwest::Url::parse(&format!("{}/getSamples", source.url)).expect("constant source URL");
    url.query_pairs_mut().extend_pairs([
        ("f", "json"),
        ("geometryType", "esriGeometryMultipoint"),
        (
            "geometry",
            &json!({"points": points, "spatialReference":{"wkid":sr}}).to_string(),
        ),
        ("returnFirstValueOnly", "true"),
        ("interpolation", "RSP_NearestNeighbor"),
    ]);
    url
}

pub(super) fn url(source: &Source, longitude: f64, latitude: f64) -> reqwest::Url {
    let sr = if source.body == "moon" {
        104903
    } else {
        104905
    };
    let stencil = super::terrain::has_stencil(source, latitude);
    let geometry = if stencil {
        json!({"points":super::terrain::points(source, longitude, latitude),"spatialReference":{"wkid":sr}})
    } else { json!({"x":longitude,"y":latitude,"spatialReference":{"wkid":sr}}) }.to_string();
    let feature = matches!(source.kind, Kind::TrekFeature);
    let base = if feature {
        format!(
            "{}/{}/query",
            source.url,
            source.layer_id.expect("feature registry layer")
        )
    } else {
        format!("{}/getSamples", source.url)
    };
    let mut url = reqwest::Url::parse(&base).expect("constant source URL");
    let mut query = url.query_pairs_mut();
    query.extend_pairs([
        ("f", "json"),
        (
            "geometryType",
            if stencil {
                "esriGeometryMultipoint"
            } else {
                "esriGeometryPoint"
            },
        ),
        ("geometry", geometry.as_str()),
    ]);
    if feature {
        query.extend_pairs([
            ("outFields", "*"),
            ("returnGeometry", "false"),
            ("spatialRel", "esriSpatialRelIntersects"),
        ]);
    } else {
        query.extend_pairs([
            ("returnFirstValueOnly", "true"),
            ("interpolation", "RSP_NearestNeighbor"),
        ]);
    }
    drop(query);
    url
}

pub(super) fn parse(
    source: &Source,
    body: &[u8],
    measurement: &mut PointMeasurement,
) -> Result<(), String> {
    let json: Value = serde_json::from_slice(body).map_err(|_| "Source returned invalid JSON")?;
    if let Some(error) = json.get("error") {
        let code = error.get("code").and_then(Value::as_u64).unwrap_or(0);
        return Err(format!("ArcGIS source error {code}"));
    }
    if matches!(source.kind, Kind::TrekFeature) {
        let features = json
            .get("features")
            .and_then(Value::as_array)
            .ok_or("Source omitted feature records")?;
        if features.is_empty() {
            measurement.status = MeasurementStatus::NoData.into();
            measurement.reason =
                "No mapped feature intersects the exact point; this does not establish absence"
                    .into();
        } else {
            let mut attrs = Vec::new();
            for feature in features.iter().take(16) {
                let attributes = feature
                    .get("attributes")
                    .filter(|a| a.is_object())
                    .ok_or("Source omitted feature attributes")?;
                attrs.push(value::from_json(attributes.clone()));
            }
            measurement.status = MeasurementStatus::Available.into();
            measurement.reason.clear();
            measurement.value = Some(value::list(attrs));
        }
        if features.len() > 16
            || json.get("exceededTransferLimit").and_then(Value::as_bool) == Some(true)
        {
            measurement
                .quality_flags
                .push("truncated_feature_matches".into());
        }
        return Ok(());
    }
    let samples = json
        .get("samples")
        .and_then(Value::as_array)
        .ok_or("Source omitted numeric samples")?;
    let sample = samples.first();
    let number = sample.and_then(|s| s.get("value")).and_then(|v| {
        v.as_f64()
            .or_else(|| v.as_str().and_then(|v| v.parse().ok()))
    });
    set_number(source, number, measurement);
    // getSamples reports resolution in source-native units, not input-geometry units.
    if let Some(spacing) = sample
        .and_then(|s| s.get("resolution"))
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite() && *v > 0.)
    {
        measurement.pixel_spacing = spacing;
    }
    Ok(())
}

pub(super) fn set_number(source: &Source, number: Option<f64>, measurement: &mut PointMeasurement) {
    if let Some(number) = number.filter(|v| source.valid(*v)) {
        measurement.status = MeasurementStatus::Available.into();
        measurement.reason.clear();
        measurement.value = Some(value::wrap(proto::value::Kind::NumberValue(number)));
    } else {
        measurement.status = MeasurementStatus::NoData.into();
        measurement.value = None;
        measurement.reason = "No valid numeric sample; source mask, missing sentinel or physical range check rejected it".into();
    }
}
