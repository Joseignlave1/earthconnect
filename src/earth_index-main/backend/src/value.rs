use crate::proto::{self, value::Kind};
use std::collections::HashMap;

pub fn null() -> proto::Value {
    wrap(Kind::NullValue(0))
}
pub fn wrap(kind: Kind) -> proto::Value {
    proto::Value { kind: Some(kind) }
}
pub fn string(v: impl Into<String>) -> proto::Value {
    wrap(Kind::StringValue(v.into()))
}
pub fn list(v: Vec<proto::Value>) -> proto::Value {
    wrap(Kind::ListValue(proto::List { values: v }))
}
pub fn from_json(v: serde_json::Value) -> proto::Value {
    match v {
        serde_json::Value::Null => null(),
        serde_json::Value::Bool(v) => wrap(Kind::BoolValue(v)),
        serde_json::Value::Number(v) => {
            if let Some(i) = v.as_i64() {
                wrap(Kind::IntegerValue(i))
            } else if v.as_u64().is_some() {
                string(v.to_string())
            } else {
                wrap(Kind::NumberValue(v.as_f64().unwrap()))
            }
        }
        serde_json::Value::String(v) => string(v),
        serde_json::Value::Array(v) => list(v.into_iter().map(from_json).collect()),
        serde_json::Value::Object(v) => wrap(Kind::ObjectValue(proto::Object {
            fields: v.into_iter().map(|(k, v)| (k, from_json(v))).collect(),
        })),
    }
}
pub fn to_json(v: &proto::Value) -> Result<serde_json::Value, String> {
    Ok(match v.kind.as_ref().ok_or("Value.kind is required")? {
        Kind::NullValue(0) => serde_json::Value::Null,
        Kind::NullValue(_) => return Err("Invalid null enum value".into()),
        Kind::BoolValue(v) => (*v).into(),
        Kind::IntegerValue(v) => (*v).into(),
        Kind::NumberValue(v) => serde_json::Number::from_f64(*v)
            .ok_or("JSON numbers must be finite")?
            .into(),
        Kind::StringValue(v) => v.clone().into(),
        Kind::BytesValue(_) => {
            return Err("Binary values cannot be nested inside JSON; bind bytes separately".into());
        }
        Kind::ListValue(v) => {
            serde_json::Value::Array(v.values.iter().map(to_json).collect::<Result<_, _>>()?)
        }
        Kind::ObjectValue(v) => serde_json::Value::Object(
            v.fields
                .iter()
                .map(|(k, v)| Ok((k.clone(), to_json(v)?)))
                .collect::<Result<_, String>>()?,
        ),
    })
}
pub fn object(v: serde_json::Value) -> proto::Object {
    if let Some(Kind::ObjectValue(o)) = from_json(v).kind {
        o
    } else {
        proto::Object {
            fields: HashMap::new(),
        }
    }
}
