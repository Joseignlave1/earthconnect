use crate::proto::{MeasurementStatus, PlanetaryBody, PointMeasurement};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize)]
pub(super) struct Source {
    pub body: String,
    pub variable_id: String,
    pub label: String,
    pub unit: String,
    pub notes: String,
    pub quality_flags: Vec<String>,
    pub source_id: String,
    pub kind: Kind,
    pub url: String,
    pub projection: Projection,
    pub extent: [f64; 4],
    pub pixel_spacing: f64,
    pub pixel_spacing_unit: String,
    pub nodata: Vec<f64>,
    pub valid_range: Option<[f64; 2]>,
    pub layer_id: Option<u32>,
    pub ppd: Option<f64>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Kind {
    TrekRaster,
    TrekFeature,
    MarsWms,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(super) enum Projection {
    Geographic,
    Polar { north: bool },
    Orthographic { longitude: f64, latitude: f64 },
    PlateCarree,
}

#[derive(Deserialize)]
struct Registry {
    sources: Vec<Source>,
}

pub(super) fn registry() -> Vec<Source> {
    serde_json::from_str::<Registry>(include_str!("sources.json"))
        .expect("reviewed planetary source registry")
        .sources
}

impl Source {
    pub fn body(&self) -> PlanetaryBody {
        match self.body.as_str() {
            "moon" => PlanetaryBody::Moon,
            "mars" => PlanetaryBody::Mars,
            _ => unreachable!("static registry body"),
        }
    }

    pub fn template(&self) -> PointMeasurement {
        PointMeasurement {
            variable_id: self.variable_id.clone(),
            label: self.label.clone(),
            unit: self.unit.clone(),
            status: MeasurementStatus::SourceTimeout.into(),
            source_id: self.source_id.clone(),
            source_url: self.url.clone(),
            pixel_spacing: self.pixel_spacing,
            pixel_spacing_unit: self.pixel_spacing_unit.clone(),
            quality_flags: self.quality_flags.clone(),
            notes: self.notes.clone(),
            reason: "Point-sampling time budget exhausted".into(),
            ..Default::default()
        }
    }

    // Conservative rectangular service-footprint check. The actual raster mask
    // can still yield NO_DATA inside this rectangle. All projected registry
    // sources are lunar, spherical, zero-offset definitions audited in sources.json.
    pub fn covers(&self, longitude: f64, latitude: f64) -> bool {
        const RADIUS: f64 = 1_737_400.0;
        let lon = longitude.to_radians();
        let lat = latitude.to_radians();
        let (x, y) = match self.projection {
            Projection::Geographic => {
                let x = if self.extent[0] >= 0. && longitude < 0. {
                    longitude + 360.
                } else {
                    longitude
                };
                (x, latitude)
            }
            Projection::Polar { north } => {
                let sign = if north { 1. } else { -1. };
                if sign * latitude < 0. {
                    return false;
                }
                let rho = 2. * RADIUS * (std::f64::consts::FRAC_PI_4 - sign * lat / 2.).tan();
                (rho * lon.sin(), -sign * rho * lon.cos())
            }
            Projection::Orthographic {
                longitude,
                latitude,
            } => {
                let lon0 = longitude.to_radians();
                let lat0 = latitude.to_radians();
                let dl = lon - lon0;
                if lat0.sin() * lat.sin() + lat0.cos() * lat.cos() * dl.cos() < 0. {
                    return false;
                }
                (
                    RADIUS * lat.cos() * dl.sin(),
                    RADIUS * (lat0.cos() * lat.sin() - lat0.sin() * lat.cos() * dl.cos()),
                )
            }
            Projection::PlateCarree => (RADIUS * lon, RADIUS * lat),
        };
        // Include exact mathematical poles and antimeridian despite tiny extent
        // rounding differences. Querying the source remains the final mask test.
        let tolerance = if matches!(self.projection, Projection::Geographic) {
            0.0001
        } else {
            0.01
        };
        x >= self.extent[0] - tolerance
            && x <= self.extent[2] + tolerance
            && y >= self.extent[1] - tolerance
            && y <= self.extent[3] + tolerance
    }

    pub fn valid(&self, value: f64) -> bool {
        value.is_finite()
            && value > -1e30
            && !self
                .nodata
                .iter()
                .any(|missing| (value - missing).abs() <= 1e-7 * missing.abs().max(1.))
            && self
                .valid_range
                .is_none_or(|[min, max]| value >= min && value <= max)
    }
}
