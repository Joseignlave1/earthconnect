//! Single-pixel numeric WMS adapter; graphic/colorized maps are never sampled.
use super::sources::Source;
use std::collections::HashMap;

pub(super) fn url(source: &Source, longitude: f64, latitude: f64) -> String {
    let ppd = source.ppd.expect("WMS registry grid density");
    let x = ((longitude + 180.) * ppd)
        .floor()
        .clamp(0., 360. * ppd - 1.);
    let y = ((90. - latitude) * ppd).floor().clamp(0., 180. * ppd - 1.);
    let west = x / ppd - 180.;
    let east = (x + 1.) / ppd - 180.;
    let north = 90. - y / ppd;
    let south = 90. - (y + 1.) / ppd;
    // The service rejects an encoded slash in FORMAT=image%2Fvicar. Registry
    // layer names are trusted constants; coordinates have already been validated.
    format!(
        "{}?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS={}&STYLES=&SRS=JMARS:4&BBOX={west},{south},{east},{north}&WIDTH=1&HEIGHT=1&FORMAT=image/vicar",
        source.url, source.source_id
    )
}

pub(super) fn decode_pixel(bytes: &[u8]) -> Result<f64, String> {
    let prefix = bytes
        .get(..bytes.len().min(80))
        .ok_or("Missing VICAR label")?;
    let prefix = std::str::from_utf8(prefix).map_err(|_| "Invalid VICAR prefix")?;
    if !prefix.starts_with("LBLSIZE=") {
        return Err("Source did not return a numeric VICAR image".into());
    }
    let label_size: usize = prefix[8..]
        .split_whitespace()
        .next()
        .ok_or("Missing VICAR label size")?
        .parse()
        .map_err(|_| "Invalid VICAR label size")?;
    let label = std::str::from_utf8(bytes.get(..label_size).ok_or("Truncated VICAR label")?)
        .map_err(|_| "Invalid VICAR label")?;
    let fields = fields(label)?;
    let integer = |key| -> Result<usize, String> {
        fields
            .get(key)
            .ok_or_else(|| format!("Missing VICAR {key}"))?
            .parse()
            .map_err(|_| format!("Invalid VICAR {key}"))
    };
    if fields.get("ORG").map(String::as_str) != Some("BSQ")
        || integer("NB")? != 1
        || integer("NS")? != 1
        || integer("NL")? != 1
        || integer("NBB")? != 0
        || integer("NLB")? != 0
    {
        return Err("Unsupported VICAR pixel layout".into());
    }
    let format = fields.get("FORMAT").ok_or("Missing VICAR FORMAT")?.as_str();
    let width = match format {
        "BYTE" => 1,
        "HALF" => 2,
        "REAL" | "FULL" => 4,
        "DOUB" => 8,
        _ => return Err("Unsupported VICAR numeric type".into()),
    };
    if integer("RECSIZE")? != width {
        return Err("Unsupported VICAR record size".into());
    }
    let pixel = bytes
        .get(label_size..)
        .and_then(|b| b.get(..width))
        .ok_or("Truncated VICAR pixel")?;
    let little = if matches!(format, "REAL" | "DOUB") {
        match fields.get("REALFMT").map(String::as_str) {
            Some("RIEEE") => true,
            Some("IEEE") => false,
            _ => return Err("Unsupported VICAR float representation".into()),
        }
    } else {
        match fields.get("INTFMT").map(String::as_str) {
            Some("LOW") => true,
            Some("HIGH") => false,
            _ => return Err("Unsupported VICAR integer representation".into()),
        }
    };
    macro_rules! decode {
        ($kind:ty) => {{
            let p = pixel.try_into().map_err(|_| "Invalid VICAR pixel width")?;
            if little {
                <$kind>::from_le_bytes(p)
            } else {
                <$kind>::from_be_bytes(p)
            }
        }};
    }
    Ok(match format {
        "BYTE" => pixel[0] as f64,
        "HALF" => decode!(i16) as f64,
        "FULL" => decode!(i32) as f64,
        "REAL" => decode!(f32) as f64,
        "DOUB" => decode!(f64),
        _ => unreachable!(),
    })
}

// VICAR labels use key=value with quoted values containing spaces. Do not split
// all whitespace: HOST/TASK metadata can include embedded spaces and '=' signs.
fn fields(label: &str) -> Result<HashMap<String, String>, String> {
    let mut rest = label.trim_matches([' ', '\0']);
    let mut result = HashMap::new();
    while !rest.is_empty() {
        let at = rest.find('=').ok_or("Invalid VICAR label field")?;
        let key = rest[..at].trim();
        rest = rest[at + 1..].trim_start();
        let value;
        if let Some(quoted) = rest.strip_prefix('\'') {
            let end = quoted.find('\'').ok_or("Unterminated VICAR label string")?;
            value = &quoted[..end];
            rest = &quoted[end + 1..];
        } else {
            let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
            value = &rest[..end];
            rest = &rest[end..];
        }
        result.insert(key.to_owned(), value.to_owned());
        rest = rest.trim_matches([' ', '\0', '\n', '\r', '\t']);
    }
    Ok(result)
}
