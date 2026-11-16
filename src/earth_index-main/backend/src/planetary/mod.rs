//! Best-effort, bounded point sampling of the reviewed public Moon/Mars sources.
mod mars;
mod sources;
mod terrain;
#[cfg(test)]
mod tests;
mod trek;

use crate::proto::{
    self, MeasurementStatus, PlanetaryBody, PointMeasurement, SamplePointRequest,
    SamplePointResponse,
};
use futures_util::{StreamExt, stream};
use reqwest::Client;
use sources::{Kind, Source};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};
use tokio::{
    sync::{Mutex as AsyncMutex, Semaphore},
    time::{Instant, timeout_at},
};
use tonic::Status;

pub const MAX_SAMPLE_TIME: Duration = Duration::from_secs(20);
const MAX_UPSTREAM_BYTES: usize = 256 * 1024;
const CACHE_CAPACITY: usize = 128;
const GOOD_TTL: Duration = Duration::from_secs(6 * 3600);
const PARTIAL_TTL: Duration = Duration::from_secs(15);

#[derive(Clone)]
pub struct Sampler(Arc<Inner>);
struct Inner {
    client: Client,
    sources: Vec<Source>,
    upstream: Semaphore,
    cache: Mutex<HashMap<Key, Slot>>,
}
#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct Key(i32, u64, u64);
struct Slot {
    accessed: Instant,
    value: Arc<AsyncMutex<Option<Cached>>>,
}
struct Cached {
    until: Instant,
    response: SamplePointResponse,
}

impl Sampler {
    pub fn new() -> Result<Self, reqwest::Error> {
        let client = Client::builder()
            .user_agent("EarthIndex/0.1 planetary-point-sampler")
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(5))
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self::with_sources(client, sources::registry()))
    }

    fn with_sources(client: Client, sources: Vec<Source>) -> Self {
        Self(Arc::new(Inner {
            client,
            sources,
            upstream: Semaphore::new(8),
            cache: Mutex::new(HashMap::new()),
        }))
    }

    pub async fn sample(
        &self,
        request: SamplePointRequest,
        budget: Duration,
    ) -> Result<SamplePointResponse, Status> {
        let body = PlanetaryBody::try_from(request.body)
            .ok()
            .filter(|b| matches!(b, PlanetaryBody::Moon | PlanetaryBody::Mars))
            .ok_or_else(|| Status::invalid_argument("body must be MOON or MARS"))?;
        if !request.longitude.is_finite()
            || !request.latitude.is_finite()
            || !(-180. ..=180.).contains(&request.longitude)
            || !(-90. ..=90.).contains(&request.latitude)
        {
            return Err(Status::invalid_argument(
                "longitude must be finite in [-180,180]; latitude finite in [-90,90]",
            ));
        }
        if budget.is_zero() {
            return Err(Status::deadline_exceeded(
                "Point-sampling deadline has elapsed",
            ));
        }
        let deadline = Instant::now() + budget.min(MAX_SAMPLE_TIME);
        // Canonicalize equivalent antimeridian / signed-zero / exact-pole points;
        // never round ordinary coordinates into a different source pixel.
        let latitude = if request.latitude == 0. {
            0.
        } else {
            request.latitude
        };
        let longitude = if latitude.abs() == 90. {
            0.
        } else if request.longitude == 180. {
            -180.
        } else if request.longitude == 0. {
            0.
        } else {
            request.longitude
        };
        let key = Key(body.into(), longitude.to_bits(), latitude.to_bits());
        let slot = {
            let mut cache = self.0.cache.lock().expect("point-cache lock");
            if !cache.contains_key(&key) && cache.len() >= CACHE_CAPACITY {
                // Keep in-flight slots so concurrent identical requests coalesce.
                if let Some(oldest) = cache
                    .iter()
                    .filter(|(_, slot)| Arc::strong_count(&slot.value) == 1)
                    .min_by_key(|(_, slot)| slot.accessed)
                    .map(|(key, _)| *key)
                {
                    cache.remove(&oldest);
                } else {
                    return Err(Status::resource_exhausted(
                        "Point sampler is busy; retry later",
                    ));
                }
            }
            let slot = cache.entry(key).or_insert_with(|| Slot {
                accessed: Instant::now(),
                value: Arc::new(AsyncMutex::new(None)),
            });
            slot.accessed = Instant::now();
            slot.value.clone()
        };
        let mut cached = timeout_at(deadline, slot.lock()).await.map_err(|_| {
            Status::deadline_exceeded("Deadline waiting for an identical point sample")
        })?;
        if let Some(entry) = cached.as_ref().filter(|entry| entry.until > Instant::now()) {
            let mut response = entry.response.clone();
            response.cached = true;
            return Ok(response);
        }
        let response = self.collect(body, longitude, latitude, deadline).await;
        let ttl = if response.partial {
            PARTIAL_TTL
        } else {
            GOOD_TTL
        };
        *cached = Some(Cached {
            until: Instant::now() + ttl,
            response: response.clone(),
        });
        Ok(response)
    }

    async fn collect(
        &self,
        body: PlanetaryBody,
        longitude: f64,
        latitude: f64,
        deadline: Instant,
    ) -> SamplePointResponse {
        let sources: Vec<_> = self.0.sources.iter().filter(|s| s.body() == body).collect();
        let mut measurements: Vec<_> = sources.iter().map(|s| s.template()).collect();
        let mut applicable = Vec::new();
        for (i, source) in sources.iter().enumerate() {
            if source.covers(longitude, latitude) {
                applicable.push((i, (*source).clone()));
            } else {
                measurements[i].status = MeasurementStatus::OutsideCoverage.into();
                measurements[i].reason = "Outside the service's rectangular footprint".into();
            }
        }
        let mut pending = stream::iter(applicable.into_iter().map(|(i, source)| {
            let sampler = self.clone();
            async move {
                (
                    i,
                    sampler.probe(&source, longitude, latitude, deadline).await,
                )
            }
        }))
        .buffer_unordered(6);
        let mut terrain_values = Vec::new();
        while let Ok(Some((i, (result, derived)))) = timeout_at(deadline, pending.next()).await {
            measurements[i] = result;
            terrain_values.extend(derived);
        }
        // Dropping the stream cancels pending HTTP operations, including semaphore
        // waits. The caller receives completed values and explicit timeout entries.
        drop(pending);
        measurements.extend(terrain_values);
        derive_temperature_range(body, &mut measurements);
        let partial = measurements.iter().any(|m| {
            matches!(
                m.status(),
                MeasurementStatus::SourceError | MeasurementStatus::SourceTimeout
            ) || m.quality_flags.iter().any(|flag| {
                flag == "truncated_feature_matches" || flag == "incomplete_terrain_window"
            })
        });
        SamplePointResponse {
            body: body.into(),
            longitude,
            latitude,
            measurements,
            partial,
            cached: false,
            retrieved_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    async fn probe(
        &self,
        source: &Source,
        longitude: f64,
        latitude: f64,
        deadline: Instant,
    ) -> (PointMeasurement, Vec<PointMeasurement>) {
        let mut measurement = source.template();
        let mut derived = Vec::new();
        let url = match source.kind {
            Kind::MarsWms => mars::url(source, longitude, latitude),
            _ => trek::url(source, longitude, latitude).to_string(),
        };
        let outcome = self.fetch(&url, matches!(source.kind, Kind::MarsWms)).await;
        match outcome {
            Ok(body) => {
                let parsed = match source.kind {
                    Kind::MarsWms => mars::decode_pixel(&body)
                        .map(|number| trek::set_number(source, Some(number), &mut measurement)),
                    _ => trek::parse(source, &body, &mut measurement),
                };
                if let Err(error) = parsed {
                    measurement.status = MeasurementStatus::SourceError.into();
                    measurement.reason = error;
                    measurement.value = None;
                } else if terrain::enabled(source) {
                    derived = terrain::derive(source, &body, latitude, &mut measurement);
                    // Leave time to return an already successful center/native
                    // sample even if the optional kilometre window stalls.
                    let window_deadline = deadline
                        .checked_sub(Duration::from_millis(50))
                        .unwrap_or(deadline)
                        .min(Instant::now() + Duration::from_secs(6));
                    derived.extend(
                        self.terrain_window(source, longitude, latitude, window_deadline)
                            .await,
                    );
                }
            }
            Err((status, reason)) => {
                measurement.status = status.into();
                measurement.reason = reason;
            }
        }
        (measurement, derived)
    }

    async fn terrain_window(
        &self,
        source: &Source,
        longitude: f64,
        latitude: f64,
        deadline: Instant,
    ) -> Vec<PointMeasurement> {
        let points = terrain::window_points(source, longitude, latitude);
        // Trek rejects large GET URIs and POSTs on these public services. Small,
        // bounded GET batches retain exact sample IDs; no whole raster download.
        let urls: Vec<_> = points
            .chunks(48)
            .enumerate()
            .map(|(batch, points)| {
                (
                    batch * 48,
                    points.len(),
                    trek::multipoint_url(source, points).to_string(),
                )
            })
            .collect();
        let mut requests = stream::iter(urls.into_iter().map(|(offset, length, url)| async move {
            (offset, length, self.fetch(&url, false).await)
        }))
        .buffer_unordered(3);
        let mut samples = Vec::new();
        while let Ok(Some((offset, length, result))) = timeout_at(deadline, requests.next()).await {
            if let Ok(bytes) = result
                && let Ok(json) = serde_json::from_slice::<serde_json::Value>(&bytes)
                && let Some(chunk) = json.get("samples").and_then(serde_json::Value::as_array)
            {
                for sample in chunk {
                    if let Some(id) = sample
                        .get("locationId")
                        .and_then(serde_json::Value::as_u64)
                        .filter(|i| *i < length as u64)
                    {
                        let mut sample = sample.clone();
                        sample["locationId"] = (id + offset as u64).into();
                        samples.push(sample);
                    }
                }
            }
        }
        terrain::derive_window(source, &serde_json::json!({"samples":samples}))
    }

    async fn fetch(
        &self,
        url: &str,
        numeric_vicar: bool,
    ) -> Result<Vec<u8>, (MeasurementStatus, String)> {
        let _permit = self
            .0
            .upstream
            .acquire()
            .await
            .expect("sampler semaphore remains open");
        let error = |e: reqwest::Error| {
            tracing::debug!(error = ?e, "Planetary public-source HTTP failure");
            if e.is_timeout() {
                (
                    MeasurementStatus::SourceTimeout,
                    "Public source timed out".into(),
                )
            } else {
                (
                    MeasurementStatus::SourceError,
                    "Public-source HTTP request failed".into(),
                )
            }
        };
        let response = self.0.client.get(url).send().await.map_err(error)?;
        if !response.status().is_success() {
            return Err((
                MeasurementStatus::SourceError,
                format!("Public source returned HTTP {}", response.status().as_u16()),
            ));
        }
        if response
            .content_length()
            .is_some_and(|n| n > MAX_UPSTREAM_BYTES as u64)
        {
            return Err((
                MeasurementStatus::SourceError,
                "Source response exceeds the point-sampling byte limit".into(),
            ));
        }
        let mut stream = response.bytes_stream();
        let mut body = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(error)?;
            if body.len() + chunk.len() > MAX_UPSTREAM_BYTES {
                return Err((
                    MeasurementStatus::SourceError,
                    "Source response exceeds the point-sampling byte limit".into(),
                ));
            }
            body.extend_from_slice(&chunk);
            // ASU closes some EOF-delimited TLS responses without close_notify.
            // A validated one-pixel VICAR label gives us an exact payload boundary;
            // stop once that complete pixel is authenticated instead of requiring
            // a trailing transport close. Truncated/malformed payloads still fail.
            if numeric_vicar && mars::decode_pixel(&body).is_ok() {
                break;
            }
        }
        Ok(body)
    }
}

fn derive_temperature_range(body: PlanetaryBody, measurements: &mut Vec<PointMeasurement>) {
    let (prefix, min_id, max_id) = match body {
        PlanetaryBody::Moon => ("moon", "moon.surface_min", "moon.surface_max"),
        PlanetaryBody::Mars => (
            "mars",
            "mars.modeled_surface_temperature_min",
            "mars.modeled_surface_temperature_max",
        ),
        _ => return,
    };
    let minimum = measurements.iter().find(|m| m.variable_id == min_id);
    let maximum = measurements.iter().find(|m| m.variable_id == max_id);
    if let (Some(minimum), Some(maximum)) = (minimum, maximum) {
        let mut range = PointMeasurement {
            variable_id: format!("{prefix}.surface_temperature_range"),
            label: "Maximum-minus-minimum surface temperature".into(),
            unit: "K_difference".into(),
            source_id: "derived_temperature_range".into(),
            status: MeasurementStatus::NoData.into(),
            pixel_spacing: maximum.pixel_spacing,
            pixel_spacing_unit: maximum.pixel_spacing_unit.clone(),
            quality_flags: vec![
                "derived_from_matching_source_extrema".into(),
                "cross_body_comparison_requires_review".into(),
            ],
            notes: format!(
                "Derived from {} and {}; no Celsius offset applied to the difference. It is not a current or instantaneous temperature range.",
                minimum.source_id, maximum.source_id
            ),
            reason: "Both matched source extrema must be available and ordered".into(),
            ..Default::default()
        };
        let number = |m: &PointMeasurement| match m.value.as_ref()?.kind.as_ref()? {
            proto::value::Kind::NumberValue(v) if m.status() == MeasurementStatus::Available => {
                Some(*v)
            }
            _ => None,
        };
        if let (Some(low), Some(high)) = (number(minimum), number(maximum))
            && high >= low
            && maximum.pixel_spacing == minimum.pixel_spacing
            && maximum.pixel_spacing_unit == minimum.pixel_spacing_unit
        {
            range.status = MeasurementStatus::Available.into();
            range.reason.clear();
            range.value = Some(crate::value::wrap(proto::value::Kind::NumberValue(
                high - low,
            )));
        }
        measurements.push(range);
    }
}
