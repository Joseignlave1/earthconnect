//! Native scalars/BLOBs and homogeneous SQL lists. Objects and heterogeneous
//! JSON lists are bound as JSON, with a typed placeholder emitted by the lexer.
use crate::{
    error::ApiError,
    proto::{Value, value::Kind},
    value,
};
use duckdb::types::Value as D;

pub fn is_json(v: &Value) -> bool {
    fn shape(v: &Value) -> Result<Option<String>, ()> {
        Ok(match &v.kind {
            Some(Kind::NullValue(_)) => None,
            Some(Kind::BoolValue(_)) => Some("bool".into()),
            Some(Kind::IntegerValue(_)) => Some("int".into()),
            Some(Kind::NumberValue(_)) => Some("double".into()),
            Some(Kind::StringValue(_)) => Some("text".into()),
            Some(Kind::BytesValue(_)) => Some("blob".into()),
            Some(Kind::ListValue(l)) => {
                let mut item = None;
                for v in &l.values {
                    if let Some(s) = shape(v)? {
                        if item.as_ref().is_some_and(|prev| prev != &s) {
                            return Err(());
                        }
                        item = Some(s);
                    }
                }
                item.map(|s| format!("list<{s}>"))
            }
            None => None,
            _ => return Err(()),
        })
    }
    shape(v).is_err()
}
pub fn bind(v: Value) -> Result<D, ApiError> {
    if is_json(&v) {
        return Ok(D::Text(
            serde_json::to_string(&value::to_json(&v).map_err(ApiError::bad_request)?)
                .map_err(|_| ApiError::bad_request("Invalid JSON parameter"))?,
        ));
    }
    Ok(
        match v
            .kind
            .ok_or_else(|| ApiError::bad_request("Value.kind is required"))?
        {
            Kind::NullValue(0) => D::Null,
            Kind::NullValue(_) => return Err(ApiError::bad_request("Invalid null enum")),
            Kind::BoolValue(v) => D::Boolean(v),
            Kind::IntegerValue(v) => D::BigInt(v),
            Kind::NumberValue(v) if v.is_finite() => D::Double(v),
            Kind::NumberValue(_) => {
                return Err(ApiError::bad_request("Parameter numbers must be finite"));
            }
            Kind::StringValue(v) => D::Text(v),
            Kind::BytesValue(v) => D::Blob(v),
            Kind::ListValue(v) => {
                D::List(v.values.into_iter().map(bind).collect::<Result<_, _>>()?)
            }
            Kind::ObjectValue(_) => unreachable!("objects bind as JSON"),
        },
    )
}
