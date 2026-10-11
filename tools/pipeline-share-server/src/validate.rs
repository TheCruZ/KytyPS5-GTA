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

/// What the emulator links a pipeline with from a module record (`M <id> <stage> ...`): its stage,
/// the vertex attributes it reads (`attr=`) and, for an LS, its patch size (`ticp=`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleInfo {
    pub stage: String,
    pub attributes: usize,
    pub control_points: u64,
}

/// The fields of a record after its kind (and, for a module, its id): `key=value` or a bare flag.
fn fields(rest: &str) -> impl Iterator<Item = (&str, Option<&str>)> {
    rest.split(' ').filter(|t| !t.is_empty()).map(|t| match t.split_once('=') {
        Some((k, v)) => (k, Some(v)),
        None => (t, None),
    })
}

/// The module of an `M` record (None for other records).
pub fn module_info(line: &str) -> Option<ModuleInfo> {
    let rest = line.strip_prefix("M ")?;
    let (_, tail) = rest.split_once(' ')?;
    let (stage, tail) = tail.split_once(' ').unwrap_or((tail, ""));
    let mut info = ModuleInfo { stage: stage.to_string(), attributes: 0, control_points: 0 };
    for (key, value) in fields(tail) {
        match (key, value) {
            ("attr", Some(v)) => info.attributes = v.split(',').count(),
            ("ticp", Some(v)) => info.control_points = v.parse().unwrap_or(u64::MAX),
            _ => {}
        }
    }
    Some(info)
}

/// Why no draw creates the pipeline of a `P` record from the modules it names, or None. As the
/// emulator's `LoadManifest()` (pipelineManifest.cpp, `PipelineLinkError()`): a VS or mesh stage,
/// or an LS, HS and TES, with an optional PS; as many vertex attributes as the VS or LS reads (none
/// for a mesh stage), each from a binding the pipeline has; the LS's patch size, or patches of 3
/// points for a rectangle list. `module` gives the modules by id; every module the record names
/// must be there (the caller waits for them).
pub fn pipeline_link_error<'a>(line: &str, module: impl Fn(&str) -> Option<&'a ModuleInfo>) -> Option<String> {
    let rest = line.strip_prefix("P ")?;
    let mut slots: [Option<&ModuleInfo>; 4] = [None; 4];
    let (mut plain, mut bindings, mut attributes) = (false, 0usize, Vec::new());
    let (mut points, mut topology) = (0u64, "PointList");
    for (key, value) in fields(rest) {
        let slot = ["v", "h", "d", "p"].iter().position(|s| *s == key);
        match (key, value, slot) {
            (_, Some(id), Some(i)) => match module(id) {
                Some(m) => slots[i] = Some(m),
                None => return Some(format!("module {id} is not known")),
            },
            ("plain", None, _) => plain = true,
            ("vb", Some(v), _) => bindings = v.split(',').count(),
            ("va", Some(v), _) => {
                for item in v.split(',') {
                    let binding = item.split_once('.').and_then(|(_, b)| b.parse::<usize>().ok());
                    match binding {
                        Some(b) => attributes.push(b),
                        None => return Some(format!("vertex attribute {item} does not read")),
                    }
                }
            }
            ("pcp", Some(v), _) => points = v.parse().unwrap_or(u64::MAX),
            ("topo", Some(v), _) => topology = v,
            _ => {}
        }
    }
    let stage = |i: usize| slots[i].map_or("-", |m| m.stage.as_str());
    let Some(vertex) = slots[0] else { return Some("no vertex stage".into()) };
    if plain && slots[3].is_none() {
        return Some("a plain pixel stage without a pixel stage".into());
    }
    let tessellation = vertex.stage == "ls";
    let mesh = vertex.stage == "mesh";
    let linked = if tessellation {
        stage(1) == "hs" && stage(2) == "tes"
    } else {
        (vertex.stage == "vs" || mesh) && slots[1].is_none() && slots[2].is_none()
    };
    if !linked || !matches!(stage(3), "-" | "ps") {
        return Some(format!("stages {},{},{},{}", stage(0), stage(1), stage(2), stage(3)));
    }
    let expected = if mesh { 0 } else { vertex.attributes };
    if attributes.len() != expected || (mesh && bindings != 0) {
        return Some(format!(
            "{} vertex attributes for a {} stage that reads {expected}",
            attributes.len(),
            vertex.stage
        ));
    }
    if let Some((i, b)) = attributes.iter().enumerate().find(|(_, b)| **b >= bindings) {
        return Some(format!("vertex attribute {i} reads binding {b} of {bindings}"));
    }
    let patches = topology == "PatchList";
    let drawn = if tessellation {
        vertex.control_points
    } else if !mesh && patches {
        3
    } else {
        0
    };
    if points != drawn || (tessellation && (points == 0 || !patches)) {
        return Some(format!("patches of {points} points where its draws have {drawn}"));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn modules() -> HashMap<&'static str, ModuleInfo> {
        [
            ("00000000000a", "M 00000000000a vs code=1 attr=c0000.1.2.9.3.0.0.0,40000.1.2.12.4.1.0.1 vbuf=12.1.0"),
            ("00000000000b", "M 00000000000b ps code=2 in=0"),
            ("00000000000c", "M 00000000000c ls code=3 ticp=3 attr=c0000.1.2.9.3.0.0.0"),
            ("00000000000d", "M 00000000000d hs code=4 ticp=3"),
            ("00000000000e", "M 00000000000e tes code=5 ticp=3"),
            ("00000000000f", "M 00000000000f mesh code=6 mt=64"),
            ("000000000010", "M 000000000010 cs code=7 t=64,1,1"),
        ]
        .into_iter()
        .map(|(id, line)| (id, module_info(line).unwrap()))
        .collect()
    }

    fn link(line: &str) -> Option<String> {
        let m = modules();
        pipeline_link_error(line, |id| m.get(id))
    }

    #[test]
    fn module_records() {
        let m = module_info("M 00000000000c ls code=3 ud=16 ticp=2 tocp=2 attr=a.1,b.2,c.3 vbuf=1.2.0").unwrap();
        assert_eq!(m, ModuleInfo { stage: "ls".into(), attributes: 3, control_points: 2 });
        let m = module_info("M 00000000000b ps").unwrap();
        assert_eq!(m, ModuleInfo { stage: "ps".into(), attributes: 0, control_points: 0 });
        assert!(module_info("P v=00000000000a").is_none());
    }

    #[test]
    fn linkable_pipelines() {
        for good in [
            "P v=00000000000a p=00000000000b plain cc=1 topo=TriangleList vb=32.0 va=0.0,16.0",
            "P v=00000000000a depth=D32Sfloat vb=12.0,4.1 va=0.0,0.1",
            // A rectangle list draws patches of 3 points through generated stages.
            "P v=00000000000a p=00000000000b topo=PatchList pcp=3 vb=32.0 va=0.0,16.0",
            "P v=00000000000c h=00000000000d d=00000000000e p=00000000000b topo=PatchList pcp=3 vb=100.0 va=0.0",
            "P v=00000000000f p=00000000000b",
            "P v=00000000000f topo=PatchList",
        ] {
            assert_eq!(link(good), None, "{good}");
        }
    }

    #[test]
    fn unlinkable_pipelines() {
        for (bad, why) in [
            // Elden Ring Nightreign: a TES as the vertex stage (a recording that mixed up modules).
            ("P v=00000000000e p=00000000000b plain vb=12.0,4.0 va=0.0,0.1", "stages tes,-,-,ps"),
            ("P v=00000000000b p=00000000000b", "stages ps,-,-,ps"),
            ("P v=00000000000a p=00000000000a vb=32.0 va=0.0,16.0", "stages vs,-,-,vs"),
            ("P v=000000000010 p=00000000000b", "stages cs,-,-,ps"),
            ("P v=00000000000a h=00000000000d d=00000000000e vb=32.0 va=0.0,16.0", "stages vs,hs,tes,-"),
            ("P v=00000000000c h=00000000000b d=00000000000e topo=PatchList pcp=3 vb=1.0 va=0.0", "stages ls,ps,tes,-"),
            ("P v=00000000000a vb=32.0 va=0.0", "1 vertex attributes for a vs stage that reads 2"),
            ("P v=00000000000a vb=32.0 va=0.0,16.0,20.0", "3 vertex attributes for a vs stage that reads 2"),
            ("P v=00000000000a vb=32.0 va=0.0,16.1", "vertex attribute 1 reads binding 1 of 1"),
            ("P v=00000000000a va=0.0,16.0", "vertex attribute 0 reads binding 0 of 0"),
            ("P v=00000000000a vb=32.0 va=0.0,x", "vertex attribute x does not read"),
            ("P v=00000000000f vb=4.0", "0 vertex attributes for a mesh stage that reads 0"),
            ("P v=00000000000a topo=PatchList vb=32.0 va=0.0,16.0", "patches of 0 points where its draws have 3"),
            ("P v=00000000000a pcp=1 vb=32.0 va=0.0,16.0", "patches of 1 points where its draws have 0"),
            ("P v=00000000000c h=00000000000d d=00000000000e topo=PatchList pcp=1 vb=1.0 va=0.0", "patches of 1 points where its draws have 3"),
            ("P v=00000000000c h=00000000000d d=00000000000e pcp=3 vb=1.0 va=0.0", "patches of 3 points where its draws have 3"),
            ("P p=00000000000b", "no vertex stage"),
            ("P v=00000000000a plain vb=32.0 va=0.0,16.0", "a plain pixel stage without a pixel stage"),
            ("P v=00000000000a p=0000000000ff", "module 0000000000ff is not known"),
        ] {
            assert_eq!(link(bad).as_deref(), Some(why), "{bad}");
        }
    }

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
