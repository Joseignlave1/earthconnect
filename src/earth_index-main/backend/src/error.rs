use tonic::{Code, Status};

#[derive(Debug)]
pub struct ApiError {
    pub status: Code,
    pub code: String,
    pub message: String,
    pub sqlstate: Option<String>,
}
impl ApiError {
    pub fn new(status: Code, code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            status,
            code: code.into(),
            message: message.into(),
            sqlstate: None,
        }
    }
    pub fn bad_request(message: impl Into<String>) -> Self {
        Self::new(Code::InvalidArgument, "INVALID_REQUEST", message)
    }
    pub fn unavailable() -> Self {
        Self::new(
            Code::Unavailable,
            "DATABASE_UNAVAILABLE",
            "Database is unavailable",
        )
    }
}
impl From<duckdb::Error> for ApiError {
    fn from(error: duckdb::Error) -> Self {
        let message = error.to_string();
        let (status, code) = if message.contains("Interrupt") {
            (Code::Cancelled, "QUERY_CANCELLED")
        } else if message.contains("Permission")
            || message.contains("read-only")
            || message.contains("disabled by configuration")
        {
            (Code::PermissionDenied, "READ_ONLY")
        } else if message.contains("Out of Memory") {
            (Code::ResourceExhausted, "DATABASE_MEMORY_LIMIT")
        } else {
            (Code::InvalidArgument, "SQL_ERROR")
        };
        Self::new(status, code, message)
    }
}
impl From<ApiError> for Status {
    fn from(error: ApiError) -> Self {
        let mut status = Status::new(error.status, error.message);
        status.metadata_mut().insert(
            "error-code",
            error.code.parse().expect("ASCII internal error code"),
        );
        if let Some(sqlstate) = error.sqlstate {
            status
                .metadata_mut()
                .insert("sqlstate", sqlstate.parse().expect("ASCII SQLSTATE"));
        }
        status
    }
}
