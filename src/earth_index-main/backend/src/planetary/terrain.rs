//! Local DEM estimates retained for the map demo, not kilometre-cell statistics.
use super::{
    sources::{Kind, Source},
    trek,
};
use crate::proto::{MeasurementStatus, PointMeasurement};
use serde_json::Value;

pub(super) fn enabled(source: &Source) -> bool {
    matches!(source.kind, Kind::TrekRaster) && source.variable_id.ends_with(".elevation")
}
pub(super) fn has_stencil(source: &Source, latitude: f64) -> bool {
    enabled(source) && latitude.abs() + source.pixel_spacing <= 90.
}
pub(super) fn points(source: &Source, longitude: f64, latitude: f64) -> Vec<[f64; 2]> {
    (0..3)
        .flat_map(|row| {
            (0..3).map(move |col| {
                [
                    (longitude + (col as f64 - 1.) * source.pixel_spacing + 180.).rem_euclid(360.)
                        - 180.,
                    latitude + (1. - row as f64) * source.pixel_spacing,
                ]
            })
        })
        .collect()
}
pub(super) fn derive(
    source: &Source,
    bytes: &[u8],
    latitude: f64,
    elevation: &mut PointMeasurement,
) -> Vec<PointMeasurement> {
    let mut heights = [None; 9];
    if has_stencil(source, latitude)
        && let Ok(json) = serde_json::from_slice::<Value>(bytes)
        && let Some(samples) = json.get("samples").and_then(Value::as_array)
    {
        for sample in samples {
            if let Some(i) = sample
                .get("locationId")
                .and_then(Value::as_u64)
                .filter(|i| *i < 9)
            {
                heights[i as usize] = sample
                    .get("value")
                    .and_then(|v| v.as_f64().or_else(|| v.as_str()?.parse().ok()))
                    .filter(|v| source.valid(*v));
            }
        }
        trek::set_number(source, heights[4], elevation);
    }
    let values = if heights.iter().all(Option::is_some) {
        let h = heights.map(Option::unwrap);
        let radius = if source.body == "moon" {
            1_737_400.
        } else {
            3_396_190.
        };
        let dy = radius * source.pixel_spacing.to_radians();
        let dx = dy * latitude.to_radians().cos();
        let slope = if dx > 0.01 {
            let east = (h[2] + 2. * h[5] + h[8] - h[0] - 2. * h[3] - h[6]) / (8. * dx);
            let south = (h[6] + 2. * h[7] + h[8] - h[0] - 2. * h[1] - h[2]) / (8. * dy);
            Some(east.hypot(south).atan().to_degrees())
        } else {
            None
        };
        [
            slope,
            Some(
                h.into_iter().fold(f64::NEG_INFINITY, f64::max)
                    - h.into_iter().fold(f64::INFINITY, f64::min),
            ),
        ]
    } else {
        [None, None]
    };
    [("local_slope", "Local DEM slope", "degree"), ("local_relief", "Local DEM relief", "m")].into_iter().zip(values).map(|((id,label,unit), number)| {
        let mut m = source.template();
        m.variable_id = format!("{}.{id}",source.body); m.label = label.into(); m.unit = unit.into();
        m.quality_flags = vec!["local_terrain_estimate".into(),"spatial_support_differs_from_earth_cell".into()];
        m.notes = "Horn slope / max-minus-min relief from a 3×3 native DEM stencil. Not a cell-mean slope, 5 km relief, or landing-safety estimate.".into();
        if let Some(number) = number { m.value=Some(crate::value::wrap(crate::proto::value::Kind::NumberValue(number))); m.status=MeasurementStatus::Available.into(); m.reason.clear(); }
        else { m.status=MeasurementStatus::NoData.into(); m.reason="A complete native DEM stencil is unavailable".into(); }
        m
    }).collect()
}

// A square tangent-plane grid, mapped by the spherical exponential map. This
// remains well-defined at the poles and across the date line on either body.
pub(super) fn window_points(source: &Source, longitude: f64, latitude: f64) -> Vec<[f64; 2]> {
    let radius: f64 = if source.body == "moon" {
        1_737_400.
    } else {
        3_396_190.
    };
    let lat = latitude.to_radians();
    let lon = longitude.to_radians();
    (-10..=10)
        .flat_map(|row| {
            (-10..=10).map(move |col| {
                let east = f64::from(col) * 1000.;
                let north = -f64::from(row) * 1000.;
                let distance = east.hypot(north);
                if distance == 0. {
                    return [longitude, latitude];
                }
                let angle = distance / radius;
                let step = angle.sin() / distance;
                let x = lat.cos() * lon.cos() * angle.cos()
                    + (-east * lon.sin() - north * lat.sin() * lon.cos()) * step;
                let y = lat.cos() * lon.sin() * angle.cos()
                    + (east * lon.cos() - north * lat.sin() * lon.sin()) * step;
                let z = lat.sin() * angle.cos() + north * lat.cos() * step;
                let phi = z.atan2(x.hypot(y));
                let lambda = y.atan2(x);
                // Sub-metre precision keeps batched GET URLs below proxy limits.
                let round = |v: f64| (v * 1e6).round() / 1e6;
                [
                    round((lambda.to_degrees() + 180.).rem_euclid(360.) - 180.),
                    round(phi.to_degrees()),
                ]
            })
        })
        .collect()
}

pub(super) fn derive_window(source: &Source, samples: &Value) -> Vec<PointMeasurement> {
    let mut heights = [None; 441];
    if let Some(samples) = samples.get("samples").and_then(Value::as_array) {
        for sample in samples {
            if let Some(i) = sample
                .get("locationId")
                .and_then(Value::as_u64)
                .filter(|i| *i < 441)
            {
                heights[i as usize] = sample
                    .get("value")
                    .and_then(|v| v.as_f64().or_else(|| v.as_str()?.parse().ok()))
                    .filter(|v| source.valid(*v));
            }
        }
    }
    let mut output = Vec::new();
    let center = heights[220];
    let complete = heights.iter().all(Option::is_some);
    let slope = [
        heights[198],
        heights[199],
        heights[200],
        heights[219],
        heights[221],
        heights[240],
        heights[241],
        heights[242],
    ];
    let slope = if slope.iter().all(Option::is_some) && center.is_some() {
        let h = slope.map(Option::unwrap);
        let east = (h[2] + 2. * h[4] + h[7] - h[0] - 2. * h[3] - h[5]) / 8000.;
        let south = (h[5] + 2. * h[6] + h[7] - h[0] - 2. * h[1] - h[2]) / 8000.;
        Some(east.hypot(south).atan().to_degrees())
    } else {
        None
    };
    output.push(window_measurement(
        source,
        "slope_1km",
        "1 km DEM slope",
        "degree",
        slope,
        complete,
    ));
    for width in [5_usize, 21] {
        let start = 10 - width / 2;
        let values: Vec<f64> = (start..start + width)
            .flat_map(|row| (start..start + width).filter_map(move |col| heights[row * 21 + col]))
            .collect();
        let enough = values.len() * 10 >= width * width * 9 && center.is_some();
        let relief = enough.then(|| {
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - values.iter().copied().fold(f64::INFINITY, f64::min)
        });
        let tpi =
            enough.then(|| center.unwrap() - values.iter().sum::<f64>() / values.len() as f64);
        output.push(window_measurement(
            source,
            &format!("relief_{width}km"),
            &format!("{width} km relief"),
            "m",
            relief,
            values.len() == width * width,
        ));
        output.push(window_measurement(
            source,
            &format!("tpi_{width}km"),
            &format!("{width} km terrain position"),
            "m",
            tpi,
            values.len() == width * width,
        ));
    }
    output
}
fn window_measurement(
    source: &Source,
    id: &str,
    label: &str,
    unit: &str,
    number: Option<f64>,
    complete: bool,
) -> PointMeasurement {
    let mut m = source.template();
    m.variable_id = format!("{}.{id}", source.body);
    m.label = label.into();
    m.unit = unit.into();
    m.pixel_spacing = 1000.;
    m.pixel_spacing_unit = "m".into();
    m.quality_flags = vec![
        "kilometre_terrain_estimate".into(),
        "point_samples_not_cell_averages".into(),
    ];
    if !complete {
        m.quality_flags.push("incomplete_terrain_window".into());
    }
    m.notes = "Native DEM sampled on a tangent-plane 21×21 grid at 1 km spacing. Horn slope; 5×5/21×21 max-minus-min relief and center-minus-mean terrain position, requiring >=90% valid samples and a valid center. Comparable scale to Earth context, but point samples differ from area-aggregated Earth cells; not landing-site validation.".into();
    if let Some(number) = number {
        m.value = Some(crate::value::wrap(crate::proto::value::Kind::NumberValue(
            number,
        )));
        m.status = MeasurementStatus::Available.into();
        m.reason.clear();
    } else {
        m.value = None;
        m.status = MeasurementStatus::NoData.into();
        m.reason = "Insufficient valid kilometre-grid samples".into();
    }
    m
}
