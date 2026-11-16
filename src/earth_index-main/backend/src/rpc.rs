use crate::{AppState, db, metadata, proto, ready};
use std::{sync::Arc, time::Duration};
use tonic::{Request, Response, Status};

#[derive(Clone)]
pub struct EarthIndexService(pub Arc<AppState>);

#[tonic::async_trait]
impl proto::earth_index_server::EarthIndex for EarthIndexService {
    async fn sample_point(
        &self,
        request: Request<proto::SamplePointRequest>,
    ) -> Result<Response<proto::SamplePointResponse>, Status> {
        let budget = request_timeout(&request)?.unwrap_or(crate::planetary::MAX_SAMPLE_TIME);
        let response = self
            .0
            .planetary
            .sample(request.into_inner(), budget)
            .await?;
        Ok(Response::new(response))
    }
    async fn query(
        &self,
        request: Request<proto::QueryRequest>,
    ) -> Result<Response<proto::QueryResponse>, Status> {
        let timeout = request_timeout(&request)?;
        let request = request.into_inner();
        if request.datasource_id.is_empty() {
            return Err(Status::invalid_argument(
                "datasource_id is required; call Metadata to discover datasources",
            ));
        }
        let mut state = (*self.0).clone();
        if let Some(timeout) = timeout {
            if timeout.is_zero() {
                return Err(Status::deadline_exceeded("Request deadline has elapsed"));
            }
            let caller_ms = timeout.as_millis().max(1).min(u64::MAX as u128) as u64;
            state.statement_timeout_ms = match state.statement_timeout_ms {
                0 => caller_ms,
                server_ms => server_ms.min(caller_ms),
            };
        }
        let datasource_id = request.datasource_id.clone();
        Ok(Response::new(
            db::execute(&state, &datasource_id, request)
                .await
                .map_err(Status::from)?,
        ))
    }
    async fn metadata(
        &self,
        request: Request<proto::MetadataRequest>,
    ) -> Result<Response<proto::MetadataResponse>, Status> {
        let timeout = request_timeout(&request)?.unwrap_or(Duration::from_secs(10));
        let result = tokio::time::timeout(timeout, metadata::load(&self.0))
            .await
            .map_err(|_| Status::deadline_exceeded("Metadata deadline exceeded"))?
            .map_err(Status::from)?;
        Ok(Response::new(result))
    }
    async fn health(
        &self,
        request: Request<proto::HealthRequest>,
    ) -> Result<Response<proto::HealthResponse>, Status> {
        let timeout = request_timeout(&request)?.unwrap_or(Duration::from_secs(5));
        let ready = tokio::time::timeout(timeout, ready(&self.0))
            .await
            .map_err(|_| Status::deadline_exceeded("Health deadline exceeded"))?;
        Ok(Response::new(proto::HealthResponse { ready }))
    }
}

// Axum routes do not use Tonic transport's timeout layer. Carry grpc-timeout
// into DB cancellation instead of letting an abandoned DuckDB query run.
fn request_timeout<T>(request: &Request<T>) -> Result<Option<Duration>, Status> {
    let Some(timeout) = request.metadata().get("grpc-timeout") else {
        return Ok(None);
    };
    let timeout = timeout
        .to_str()
        .map_err(|_| Status::invalid_argument("Invalid grpc-timeout"))?;
    if !(2..=9).contains(&timeout.len()) {
        return Err(Status::invalid_argument("Invalid grpc-timeout"));
    }
    let (number, unit) = timeout.split_at(timeout.len() - 1);
    if !number.bytes().all(|b| b.is_ascii_digit()) {
        return Err(Status::invalid_argument("Invalid grpc-timeout"));
    }
    let number = number
        .parse::<u64>()
        .map_err(|_| Status::invalid_argument("Invalid grpc-timeout"))?;
    Ok(Some(match unit {
        "H" => Duration::from_secs(number * 3600),
        "M" => Duration::from_secs(number * 60),
        "S" => Duration::from_secs(number),
        "m" => Duration::from_millis(number),
        "u" => Duration::from_micros(number),
        "n" => Duration::from_nanos(number),
        _ => return Err(Status::invalid_argument("Invalid grpc-timeout unit")),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn standard_deadlines() {
        for (header, want) in [
            ("50m", Duration::from_millis(50)),
            ("2S", Duration::from_secs(2)),
            ("1u", Duration::from_micros(1)),
        ] {
            let mut r = Request::new(());
            r.metadata_mut()
                .insert("grpc-timeout", header.parse().unwrap());
            assert_eq!(request_timeout(&r).unwrap(), Some(want));
        }
        for header in ["9", "-1S", "1X", "123456789S"] {
            let mut r = Request::new(());
            r.metadata_mut()
                .insert("grpc-timeout", header.parse().unwrap());
            assert!(request_timeout(&r).is_err());
        }
    }

    #[tokio::test]
    async fn deadline_includes_waiting_for_a_database_connection() {
        use proto::earth_index_server::EarthIndex;
        let database = crate::db::Database::test_database();
        let held = database
            .permits
            .clone()
            .acquire_many_owned(2)
            .await
            .unwrap();
        let service = EarthIndexService(Arc::new(AppState {
            database,
            statement_timeout_ms: 10000,
            max_response_bytes: 1 << 20,
            planetary: crate::planetary::Sampler::new().unwrap(),
        }));
        let mut request = Request::new(proto::QueryRequest {
            query: "SELECT 1".into(),
            params: Default::default(),
            datasource_id: "test".into(),
        });
        request.set_timeout(Duration::from_millis(30));
        let start = std::time::Instant::now();
        let error = service.query(request).await.unwrap_err();
        assert_eq!(error.code(), tonic::Code::DeadlineExceeded);
        assert!(start.elapsed() < Duration::from_millis(500));
        drop(held);
    }
}
