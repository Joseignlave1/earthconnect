//! Rewrite named placeholders without touching quoted SQL or comments.
use crate::{error::ApiError, proto};
use std::collections::{HashMap, HashSet};

pub struct NamedQuery {
    pub sql: String,
    pub names: Vec<String>,
}

pub fn rewrite(
    input: &str,
    params: &HashMap<String, proto::Value>,
) -> Result<NamedQuery, ApiError> {
    let b = input.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut names = Vec::<String>::new();
    let mut i = 0;
    let mut first_word = None;
    let mut ended = false;
    while i < b.len() {
        let start = i;
        if ended
            && !b[i].is_ascii_whitespace()
            && !b[i..].starts_with(b"--")
            && !b[i..].starts_with(b"/*")
        {
            return Err(ApiError::bad_request(
                "Exactly one SQL statement is allowed",
            ));
        }
        if b[i..].starts_with(b"--") {
            i += 2;
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if b[i..].starts_with(b"/*") {
            i += 2;
            let mut depth = 1;
            while i < b.len() && depth > 0 {
                if b[i..].starts_with(b"/*") {
                    depth += 1;
                    i += 2;
                } else if b[i..].starts_with(b"*/") {
                    depth -= 1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
            if depth != 0 {
                return Err(ApiError::bad_request("Unterminated SQL comment"));
            }
        } else if b[i] == b'\'' || b[i] == b'"' {
            let quote = b[i];
            let escaped = quote == b'\''
                && i > 0
                && matches!(b[i - 1], b'e' | b'E')
                && (i < 2 || !ident(b[i - 2]));
            i += 1;
            let mut closed = false;
            while i < b.len() {
                if escaped && b[i] == b'\\' {
                    i = (i + 2).min(b.len());
                } else if b[i] == quote {
                    i += 1;
                    if i < b.len() && b[i] == quote {
                        i += 1;
                    } else {
                        closed = true;
                        break;
                    }
                } else {
                    i += 1;
                }
            }
            if !closed {
                return Err(ApiError::bad_request("Unterminated SQL quote"));
            }
        } else if b[i] == b'$' && (i == 0 || !ident(b[i - 1])) {
            let mut end = i + 1;
            if end < b.len() && b[end].is_ascii_digit() {
                return Err(ApiError::bad_request(
                    "Use named :parameters instead of positional $parameters",
                ));
            }
            if end < b.len() && ident_start(b[end]) {
                end += 1;
                while end < b.len() && ident_plain(b[end]) {
                    end += 1;
                }
            }
            if end < b.len() && b[end] == b'$' {
                let delimiter = &input[i..=end];
                i = end + 1;
                let close = input[i..].find(delimiter).ok_or_else(|| {
                    ApiError::bad_request("Unterminated dollar-quoted SQL string")
                })?;
                i += close + delimiter.len();
            } else {
                return Err(ApiError::bad_request("Use named :parameters"));
            }
        } else if b[i..].starts_with(b"::") || b[i..].starts_with(b":=") {
            i += 2;
        } else if b[i] == b':' && i + 1 < b.len() && ident_start(b[i + 1]) {
            i += 2;
            while i < b.len() && ident_plain(b[i]) {
                i += 1;
            }
            let name = &input[start + 1..i];
            if !params.contains_key(name) {
                return Err(ApiError::bad_request(format!("Missing parameter :{name}")));
            }
            let n = if let Some(n) = names.iter().position(|x| x == name) {
                n + 1
            } else {
                names.push(name.into());
                names.len()
            };
            let placeholder = if crate::params::is_json(&params[name]) {
                format!("CAST(${n} AS JSON)")
            } else {
                format!("${n}")
            };
            out.extend_from_slice(placeholder.as_bytes());
            continue;
        } else if b[i] == b';' {
            ended = true;
            i += 1;
        } else if b[i] == b'?' {
            return Err(ApiError::bad_request("Use named :parameters"));
        } else if ident_start(b[i]) {
            i += 1;
            while i < b.len() && ident(b[i]) {
                i += 1;
            }
            if first_word.is_none() {
                first_word = Some(input[start..i].to_ascii_uppercase());
            }
        } else {
            i += 1;
        }
        out.extend_from_slice(&b[start..i]);
    }
    // Transaction/session commands are not query operations. DuckDB also runs
    // inside a read-only transaction with locked configuration and no external access.
    if !matches!(
        first_word.as_deref(),
        Some("SELECT" | "WITH" | "VALUES" | "TABLE" | "SHOW" | "EXPLAIN")
    ) {
        return Err(ApiError::bad_request(
            "Expected a read-only query (SELECT, WITH, VALUES, TABLE, SHOW or EXPLAIN)",
        ));
    }
    let used: HashSet<_> = names.iter().collect();
    let mut extra: Vec<_> = params.keys().filter(|k| !used.contains(k)).collect();
    extra.sort();
    if !extra.is_empty() {
        return Err(ApiError::bad_request(format!(
            "Unused parameters: {extra:?}"
        )));
    }
    Ok(NamedQuery {
        sql: String::from_utf8(out).expect("copying UTF-8 SQL"),
        names,
    })
}
fn ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}
fn ident_plain(b: u8) -> bool {
    ident_start(b) || b.is_ascii_digit()
}
fn ident(b: u8) -> bool {
    ident_plain(b) || b == b'$' || b >= 128
}

#[cfg(test)]
mod tests {
    use super::*;
    fn params() -> HashMap<String, proto::Value> {
        [
            ("one".into(), proto::Value::default()),
            ("two".into(), proto::Value::default()),
        ]
        .into()
    }
    #[test]
    fn lexical_contexts() {
        let sql = "SELECT :one::int, :two, :one, ':ignore', \"a:ignore\", $$:ignore$$, $tag$:ignore$tag$, E'\\\':ignore' /* :ignore /* nested */ */ -- :ignore\n";
        let q = rewrite(sql, &params()).unwrap();
        assert!(q.sql.starts_with("SELECT $1::int, $2, $1,"));
        assert!(q.sql.contains("$tag$:ignore$tag$"));
        assert_eq!(q.names, vec!["one", "two"]);
    }
    #[test]
    fn rejects_bad_bindings_and_session_control() {
        for sql in [
            "SELECT :missing",
            "SELECT 1",
            "SET search_path=public",
            "COMMIT",
            "SELECT $1",
            "SELECT $name",
            "SELECT ?",
            "SELECT 1; SET enable_external_access=true",
            "SELECT 'oops",
            "SELECT /*oops",
        ] {
            assert!(rewrite(sql, &params()).is_err(), "{sql}");
        }
    }
    #[test]
    fn keeps_unicode_and_identifier_dollars() {
        let q = rewrite("SELECT 'ñ :no', foo$bar, :one,:two", &params()).unwrap();
        assert!(q.sql.contains("foo$bar"));
        assert!(q.sql.contains("ñ :no"));
    }
}
