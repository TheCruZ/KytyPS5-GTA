//! Validation of the fields and records of a sync request (protocol v1).

pub const FORMAT_PREFIX: &str = "# kyty pipeline manifest v";
pub const MAX_RECORD_BYTES: usize = 8192;
pub const MAX_RECORDS_PER_REQUEST: usize = 5000;
pub const MAX_BODY_BYTES: usize = 4 * 1024 * 1024;

fn name_chars_ok(s: &str) -> bool {
    s.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// `^[A-Z]{4}[0-9]{5}$`
pub fn valid_title(s: &str) -> bool {
    let b = s.as_bytes();
    b.len() == 9 && b[..4].iter().all(u8::is_ascii_uppercase) && b[4..].iter().all(u8::is_ascii_digit)
}

/// `^[0-9A-Za-z._-]{1,32}$`
pub fn valid_version(s: &str) -> bool {
    (1..=32).contains(&s.len()) && name_chars_ok(s)
}

/// Starts with the manifest header prefix, at most 64 bytes. The part after the prefix (the
/// number of the format) must also be `[0-9A-Za-z._-]+`: it names a directory of the export.
pub fn valid_format(s: &str) -> bool {
    s.len() <= 64
        && s.strip_prefix(FORMAT_PREFIX).is_some_and(|rest| !rest.is_empty() && name_chars_ok(rest))
}

/// Directory of the export for a (valid) format: what follows the header prefix, as `v<rest>`.
pub fn format_dir(format: &str) -> String {
    format!("v{}", format.strip_prefix(FORMAT_PREFIX).unwrap_or(format))
}

/// A UUID v4 in its canonical lowercase/uppercase hyphenated form.
pub fn valid_client(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() != 36 {
        return false;
    }
    for (i, c) in b.iter().enumerate() {
        let ok = match i {
            8 | 13 | 18 | 23 => *c == b'-',
            14 => *c == b'4',
            19 => matches!(*c, b'8' | b'9' | b'a' | b'b' | b'A' | b'B'),
            _ => c.is_ascii_hexdigit(),
        };
        if !ok {
            return false;
        }
    }
    true
}

pub fn valid_emulator(s: &str) -> bool {
    s.chars().count() <= 64 && !s.chars().any(char::is_control)
}

fn is_hex12(s: &str) -> bool {
    s.len() == 12 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Module,
    Pipeline,
    Tessellation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parsed {
    pub kind: Kind,
    /// Module id (12 hex digits) of an `M` record.
    pub id: Option<String>,
    /// Modules a `P` record names through `v=`, `h=`, `d=` and `p=`.
    pub deps: Vec<String>,
}

/// Checks one record: `M `, `P ` or `T ` prefix, printable ASCII only, at most 8192 bytes. An `M`
/// carries a 12 hex digit id; the `v=`/`h=`/`d=`/`p=` fields of a `P` hold ids of that form.
pub fn parse_record(line: &str) -> Option<Parsed> {
    if line.len() > MAX_RECORD_BYTES || !line.bytes().all(|b| (0x20..=0x7e).contains(&b)) {
        return None;
    }
    let mut parsed = Parsed { kind: Kind::Module, id: None, deps: Vec::new() };
    if let Some(rest) = line.strip_prefix("M ") {
        let (id, tail) = rest.split_once(' ')?;
        if !is_hex12(id) || tail.is_empty() {
            return None;
        }
        parsed.id = Some(id.to_string());
    } else if let Some(rest) = line.strip_prefix("P ") {
        parsed.kind = Kind::Pipeline;
        for token in rest.split(' ') {
            if let Some((key, value)) = token.split_once('=') {
                if matches!(key, "v" | "h" | "d" | "p") {
                    if !is_hex12(value) {
                        return None;
                    }
                    parsed.deps.push(value.to_string());
                }
            }
        }
    } else if let Some(rest) = line.strip_prefix("T ") {
        if rest.is_empty() {
            return None;
        }
        parsed.kind = Kind::Tessellation;
    } else {
        return None;
    }
    Some(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn titles_and_versions() {
        assert!(valid_title("PPSA21564"));
        for bad in ["ppsa21564", "PPSA2156", "PPSA215644", "PPS421564", "PPSA2156a", ""] {
            assert!(!valid_title(bad), "{bad}");
        }
        assert!(valid_version("01.007.000"));
        assert!(valid_version("a_b-C.1"));
        assert!(!valid_version(""));
        assert!(!valid_version("../x"));
        assert!(!valid_version(&"1".repeat(33)));
        assert!(valid_version(&"1".repeat(32)));
    }

    #[test]
    fn formats() {
        assert!(valid_format("# kyty pipeline manifest v1"));
        assert!(!valid_format("# kyty pipeline manifest v"));
        assert!(!valid_format("# other"));
        assert!(!valid_format("# kyty pipeline manifest v1/../x"));
        assert!(!valid_format(&format!("{FORMAT_PREFIX}{}", "1".repeat(64))));
        assert_eq!(format_dir("# kyty pipeline manifest v1"), "v1");
    }

    #[test]
    fn clients() {
        assert!(valid_client("0b1c2d3e-4f50-4a6b-8c7d-9e0f1a2b3c4d"));
        assert!(!valid_client("0b1c2d3e-4f50-1a6b-8c7d-9e0f1a2b3c4d"));
        assert!(!valid_client("0b1c2d3e-4f50-4a6b-cc7d-9e0f1a2b3c4d"));
        assert!(!valid_client("not a uuid"));
    }

    #[test]
    fn records() {
        let m = parse_record("M 2386097ea6a9 vs code=00281bc8616b3c71 ud=24").unwrap();
        assert_eq!(m.kind, Kind::Module);
        assert_eq!(m.id.as_deref(), Some("2386097ea6a9"));
        let p = parse_record("P v=00349a84959a p=303523f11c68 plain vb=36.0").unwrap();
        assert_eq!(p.deps, ["00349a84959a", "303523f11c68"]);
        assert_eq!(parse_record("T 2c500da413b2bad2 icp=3").unwrap().kind, Kind::Tessellation);
        for bad in [
            "X abc",
            "M",
            "M zz vs",
            "M 2386097ea6a9",
            "M 2386097ea6a9 ",
            "P v=xyz",
            "T ",
            "M 2386097ea6a9 vs \u{e9}",
            "M 2386097ea6a9 vs\ttab",
            "",
        ] {
            assert!(parse_record(bad).is_none(), "{bad:?}");
        }
        let long = format!("T {}", "a".repeat(MAX_RECORD_BYTES - 2));
        assert!(parse_record(&long).is_some());
        assert!(parse_record(&format!("{long}a")).is_none());
    }
}
