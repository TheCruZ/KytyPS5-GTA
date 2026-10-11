//! In-memory state: records per partition, quorum, publication order and persistence.

use crate::validate::{self, Parsed};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Pairs (client, ip) kept per pending record, and per client within them.
const MAX_PAIRS: usize = 32;
const MAX_PAIRS_PER_CLIENT: usize = 2;
const STATE_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PartKey {
    pub format: String,
    pub title: String,
    pub version: String,
}

impl PartKey {
    pub fn new(format: &str, title: &str, version: &str) -> Self {
        Self { format: format.into(), title: title.into(), version: version.into() }
    }
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub quorum: usize,
    pub partition_cap: usize,
    pub max_partitions: usize,
    /// Records per response.
    pub page_size: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self { quorum: 2, partition_cap: 200_000, max_partitions: 500, page_size: 20_000 }
    }
}

/// What the repository already has for a partition: not served, not counted as new, and its
/// modules satisfy the pipelines.
#[derive(Default)]
pub struct Known {
    lines: HashSet<String>,
    modules: HashSet<String>,
}

impl Known {
    pub fn add(&mut self, line: &str) {
        if let Some(parsed) = validate::parse_record(line) {
            if let Some(id) = parsed.id {
                self.modules.insert(id);
            }
            self.lines.insert(line.to_string());
        }
    }

    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}

struct Pending {
    pairs: Vec<(u64, u64)>,
    /// Reached quorum but waits for the modules of a pipeline.
    ready: bool,
}

enum Entry {
    Published,
    Pending(Pending),
}

#[derive(Default)]
struct Partition {
    published: Vec<Arc<str>>,
    index: HashMap<Arc<str>, Entry>,
    modules: HashSet<String>,
    waiting: Vec<Arc<str>>,
    dirty: bool,
}

fn augment(c: usize, adj: &[Vec<usize>], owner: &mut [Option<usize>], seen: &mut [bool]) -> bool {
    for &i in &adj[c] {
        if seen[i] {
            continue;
        }
        seen[i] = true;
        if owner[i].is_none_or(|o| augment(o, adj, owner, seen)) {
            owner[i] = Some(c);
            return true;
        }
    }
    false
}

/// `need` distinct clients from `need` distinct IPs: a matching of that size in the bipartite
/// graph of the (client, ip) pairs.
fn quorum_met(pairs: &[(u64, u64)], need: usize) -> bool {
    if need <= 1 {
        return !pairs.is_empty();
    }
    let mut clients: Vec<u64> = pairs.iter().map(|p| p.0).collect();
    let mut ips: Vec<u64> = pairs.iter().map(|p| p.1).collect();
    clients.sort_unstable();
    clients.dedup();
    ips.sort_unstable();
    ips.dedup();
    if clients.len() < need || ips.len() < need {
        return false;
    }
    let adj: Vec<Vec<usize>> = clients
        .iter()
        .map(|c| {
            pairs
                .iter()
                .filter(|p| p.0 == *c)
                .map(|p| ips.binary_search(&p.1).unwrap_or_default())
                .collect()
        })
        .collect();
    let mut owner = vec![None; ips.len()];
    let mut count = 0;
    for c in 0..clients.len() {
        let mut seen = vec![false; ips.len()];
        if augment(c, &adj, &mut owner, &mut seen) {
            count += 1;
            if count >= need {
                return true;
            }
        }
    }
    false
}

impl Partition {
    fn deps_ok(&self, parsed: &Parsed, known: Option<&Known>) -> bool {
        parsed
            .deps
            .iter()
            .all(|d| self.modules.contains(d) || known.is_some_and(|k| k.modules.contains(d)))
    }

    /// Assigns the next sequence number. Returns whether it was a module.
    fn publish(&mut self, line: &Arc<str>, parsed: &Parsed) -> bool {
        self.index.insert(line.clone(), Entry::Published);
        self.published.push(line.clone());
        self.dirty = true;
        if let Some(id) = &parsed.id {
            self.modules.insert(id.clone());
            return true;
        }
        false
    }

    /// Publishes the pipelines that were waiting for modules and now have them.
    fn resolve_waiting(&mut self, known: Option<&Known>) -> usize {
        let mut published = 0;
        let waiting = std::mem::take(&mut self.waiting);
        for line in waiting {
            match validate::parse_record(&line) {
                Some(parsed) if self.deps_ok(&parsed, known) => {
                    self.publish(&line, &parsed);
                    published += 1;
                }
                _ => self.waiting.push(line),
            }
        }
        published
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct SubmitOutcome {
    /// New (client, record) contributions, for the daily limits.
    pub counted: u64,
    pub published: usize,
    /// Dropped because the partition is full or the daily budget ran out.
    pub dropped: usize,
    /// The partition does not exist and no more may be created.
    pub partition_refused: bool,
}

pub struct Store {
    settings: Settings,
    salt: [u8; 16],
    parts: HashMap<PartKey, Partition>,
    known: HashMap<PartKey, Known>,
    state_dirty: bool,
}

impl Store {
    pub fn new(settings: Settings) -> Self {
        Self::with_salt(settings, rand::random())
    }

    pub fn with_salt(settings: Settings, salt: [u8; 16]) -> Self {
        Self { settings, salt, parts: HashMap::new(), known: HashMap::new(), state_dirty: false }
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    /// Salted hash of an identifier (a client id or a masked IP); the clear value is not kept.
    pub fn hash(&self, tag: &str, value: &str) -> u64 {
        let mut h = Sha256::new();
        h.update(self.salt);
        h.update(tag.as_bytes());
        h.update([0]);
        h.update(value.as_bytes());
        let d = h.finalize();
        u64::from_le_bytes(d[..8].try_into().unwrap_or_default())
    }

    pub fn partitions(&self) -> usize {
        self.parts.len()
    }

    /// (published, pending) records of a partition.
    pub fn counts(&self, key: &PartKey) -> (usize, usize) {
        self.parts.get(key).map_or((0, 0), |p| (p.published.len(), p.index.len() - p.published.len()))
    }

    /// Replaces what is known for the partitions. Pending records that are now known go away and
    /// the pipelines waiting for modules that are now known are published.
    pub fn set_known(&mut self, known: HashMap<PartKey, Known>) {
        self.known = known;
        for (key, part) in &mut self.parts {
            let k = self.known.get(key);
            if let Some(k) = k {
                let gone: Vec<Arc<str>> = part
                    .index
                    .iter()
                    .filter(|(l, e)| matches!(e, Entry::Pending(_)) && k.lines.contains(&***l))
                    .map(|(l, _)| l.clone())
                    .collect();
                for line in &gone {
                    part.index.remove(line);
                }
                part.waiting.retain(|l| !k.lines.contains(&**l));
                part.dirty |= !gone.is_empty();
                self.state_dirty |= !gone.is_empty();
            }
            let n = part.resolve_waiting(k);
            self.state_dirty |= n > 0;
        }
    }

    /// Adds the records of `client` (behind `ip`) to the partition. `budget` is how many new
    /// contributions it may still make.
    pub fn submit(
        &mut self,
        key: &PartKey,
        client: u64,
        ip: u64,
        records: &[(String, Parsed)],
        mut budget: u64,
    ) -> SubmitOutcome {
        let mut out = SubmitOutcome::default();
        if records.is_empty() || budget == 0 {
            out.dropped = records.len();
            return out;
        }
        if !self.parts.contains_key(key) && self.parts.len() >= self.settings.max_partitions {
            out.partition_refused = true;
            return out;
        }
        let known = self.known.get(key);
        let part = self.parts.entry(key.clone()).or_default();
        let quorum = self.settings.quorum;
        let cap = self.settings.partition_cap;
        let mut new_modules = false;
        for (line, parsed) in records {
            if known.is_some_and(|k| k.lines.contains(line)) {
                continue;
            }
            if budget == 0 {
                out.dropped += 1;
                continue;
            }
            let line_arc: Arc<str>;
            if let Some((k, e)) = part.index.get_key_value(line.as_str()) {
                if !matches!(e, Entry::Pending(_)) {
                    continue;
                }
                line_arc = k.clone();
                let Some(Entry::Pending(p)) = part.index.get_mut(line.as_str()) else { continue };
                let by_client = p.pairs.iter().filter(|q| q.0 == client).count();
                if p.pairs.contains(&(client, ip))
                    || p.pairs.len() >= MAX_PAIRS
                    || by_client >= MAX_PAIRS_PER_CLIENT
                {
                    continue;
                }
                p.pairs.push((client, ip));
                budget -= 1;
                out.counted += 1;
                if p.ready || !quorum_met(&p.pairs, quorum) {
                    continue;
                }
                p.ready = true;
            } else {
                if part.index.len() >= cap {
                    out.dropped += 1;
                    continue;
                }
                line_arc = Arc::from(line.as_str());
                let pairs = vec![(client, ip)];
                let ready = quorum_met(&pairs, quorum);
                part.index.insert(line_arc.clone(), Entry::Pending(Pending { pairs, ready }));
                budget -= 1;
                out.counted += 1;
                if !ready {
                    continue;
                }
            }
            // This contribution completed the quorum.
            if part.deps_ok(parsed, known) {
                new_modules |= part.publish(&line_arc, parsed);
                out.published += 1;
            } else {
                part.waiting.push(line_arc);
            }
        }
        if new_modules {
            out.published += part.resolve_waiting(known);
        }
        if out.counted > 0 {
            part.dirty = true;
            self.state_dirty = true;
        }
        out
    }

    /// Published records after `cursor`: (new cursor, records, more). A cursor beyond the
    /// published records (the server lost state) restarts from the beginning.
    pub fn fetch(&self, key: &PartKey, cursor: u64) -> (u64, Vec<Arc<str>>, bool) {
        let Some(part) = self.parts.get(key) else {
            return (0, Vec::new(), false);
        };
        let len = part.published.len();
        let from = usize::try_from(cursor).ok().filter(|c| *c <= len).unwrap_or(0);
        let to = len.min(from + self.settings.page_size);
        (to as u64, part.published[from..to].to_vec(), to < len)
    }

    // ---- Persistence ----

    pub fn is_dirty(&self) -> bool {
        self.state_dirty
    }

    /// Writes the state (`state.json`) atomically and exports the changed partitions as `.kpm`.
    pub fn save(&mut self, dir: &Path) -> io::Result<()> {
        fs::create_dir_all(dir)?;
        let mut keys: Vec<PartKey> = self.parts.keys().cloned().collect();
        keys.sort();
        {
            let partitions: Vec<PartView> = keys
                .iter()
                .map(|key| {
                    let part = &self.parts[key];
                    let mut pending: Vec<PendingView> = part
                        .index
                        .iter()
                        .filter_map(|(line, e)| match e {
                            Entry::Pending(p) => Some(PendingView { line, pairs: &p.pairs, ready: p.ready }),
                            Entry::Published => None,
                        })
                        .collect();
                    pending.sort_by(|a, b| a.line.cmp(b.line));
                    PartView {
                        format: &key.format,
                        title: &key.title,
                        version: &key.version,
                        published: &part.published,
                        pending,
                    }
                })
                .collect();
            let view = StateView { version: STATE_VERSION, salt: hex(&self.salt), partitions };
            write_atomic(&dir.join("state.json"), |w| {
                serde_json::to_writer(&mut *w, &view).map_err(io::Error::other)
            })?;
        }
        for key in &keys {
            let part = &self.parts[key];
            if !part.dirty {
                continue;
            }
            let known = self.known.get(key);
            let folder = dir.join("export").join(validate::format_dir(&key.format));
            fs::create_dir_all(&folder)?;
            let base = format!("{}_{}", key.title, key.version);
            let published: Vec<&str> = part
                .published
                .iter()
                .map(|l| &**l)
                .filter(|l| !known.is_some_and(|k| k.lines.contains(*l)))
                .collect();
            let mut pending: Vec<&str> = part
                .index
                .iter()
                .filter(|(_, e)| matches!(e, Entry::Pending(_)))
                .map(|(l, _)| &**l)
                .collect();
            pending.sort_unstable();
            for (name, lines) in [(format!("{base}.kpm"), published), (format!("{base}.pending.kpm"), pending)] {
                let path = folder.join(name);
                if lines.is_empty() {
                    let _ = fs::remove_file(&path);
                    continue;
                }
                write_atomic(&path, |w| {
                    writeln!(w, "{}", key.format)?;
                    for line in lines {
                        writeln!(w, "{line}")?;
                    }
                    Ok(())
                })?;
            }
        }
        for part in self.parts.values_mut() {
            part.dirty = false;
        }
        self.state_dirty = false;
        Ok(())
    }

    /// Loads `state.json` from `dir` (a new store when there is none).
    pub fn load(settings: Settings, dir: &Path) -> io::Result<Self> {
        let path = dir.join("state.json");
        let file = match fs::File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Self::new(settings)),
            Err(e) => return Err(e),
        };
        let state: StateFile = serde_json::from_reader(io::BufReader::new(file)).map_err(io::Error::other)?;
        if state.version != STATE_VERSION {
            return Err(io::Error::other(format!("unknown state version {}", state.version)));
        }
        let salt: [u8; 16] = unhex(&state.salt)
            .and_then(|v| v.try_into().ok())
            .ok_or_else(|| io::Error::other("bad salt in state"))?;
        let mut store = Self::with_salt(settings, salt);
        for p in state.partitions {
            let key = PartKey { format: p.format, title: p.title, version: p.version };
            let mut part = Partition::default();
            for line in p.published {
                let line: Arc<str> = Arc::from(line);
                if let Some(id) = validate::parse_record(&line).and_then(|r| r.id) {
                    part.modules.insert(id);
                }
                part.index.insert(line.clone(), Entry::Published);
                part.published.push(line);
            }
            for q in p.pending {
                let line: Arc<str> = Arc::from(q.line);
                if q.ready {
                    part.waiting.push(line.clone());
                }
                part.index.insert(line, Entry::Pending(Pending { pairs: q.pairs, ready: q.ready }));
            }
            store.parts.insert(key, part);
        }
        Ok(store)
    }
}

fn write_atomic(path: &Path, f: impl FnOnce(&mut BufWriter<fs::File>) -> io::Result<()>) -> io::Result<()> {
    let mut tmp: PathBuf = path.to_path_buf();
    let mut name = tmp.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".tmp");
    tmp.set_file_name(name);
    let mut w = BufWriter::new(fs::File::create(&tmp)?);
    f(&mut w)?;
    w.flush()?;
    w.get_ref().sync_all()?;
    drop(w);
    fs::rename(&tmp, path)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) || !s.is_ascii() {
        return None;
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).ok()).collect()
}

#[derive(Serialize)]
struct StateView<'a> {
    version: u32,
    salt: String,
    partitions: Vec<PartView<'a>>,
}

#[derive(Serialize)]
struct PartView<'a> {
    format: &'a str,
    title: &'a str,
    version: &'a str,
    published: &'a [Arc<str>],
    pending: Vec<PendingView<'a>>,
}

#[derive(Serialize)]
struct PendingView<'a> {
    line: &'a str,
    pairs: &'a [(u64, u64)],
    ready: bool,
}

#[derive(Deserialize)]
struct StateFile {
    version: u32,
    salt: String,
    partitions: Vec<PartFile>,
}

#[derive(Deserialize)]
struct PartFile {
    format: String,
    title: String,
    version: String,
    published: Vec<String>,
    pending: Vec<PendingFile>,
}

#[derive(Deserialize)]
struct PendingFile {
    line: String,
    pairs: Vec<(u64, u64)>,
    ready: bool,
}

/// Loads the `.kpm` files of a directory (`<TITLE>_<VERSION>.kpm`, as `assets/pipelines`) as
/// the records the clients already have.
pub fn load_known(dir: &Path) -> io::Result<HashMap<PartKey, Known>> {
    let mut all: HashMap<PartKey, Known> = HashMap::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        let Some(stem) = path.file_name().and_then(|n| n.to_str()).and_then(|n| n.strip_suffix(".kpm")) else {
            continue;
        };
        let Some((title, version)) = stem.split_once('_') else { continue };
        if !validate::valid_title(title) || !validate::valid_version(version) {
            continue;
        }
        let text = fs::read_to_string(&path)?;
        let mut lines = text.lines().map(|l| l.trim_end_matches('\r'));
        let Some(format) = lines.next().filter(|f| validate::valid_format(f)) else { continue };
        let known = all.entry(PartKey::new(format, title, version)).or_default();
        for line in lines {
            known.add(line);
        }
    }
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FMT: &str = "# kyty pipeline manifest v1";
    const M1: &str = "M 000000000001 vs code=aa";
    const M2: &str = "M 000000000002 ps code=bb";
    const P12: &str = "P v=000000000001 p=000000000002 plain";
    const T1: &str = "T 2c500da413b2bad2 icp=3";

    fn key() -> PartKey {
        PartKey::new(FMT, "PPSA21564", "01.007.000")
    }

    fn recs(lines: &[&str]) -> Vec<(String, Parsed)> {
        lines.iter().map(|l| (l.to_string(), validate::parse_record(l).unwrap())).collect()
    }

    fn strings(lines: &[String]) -> Vec<(String, Parsed)> {
        lines.iter().map(|l| (l.clone(), validate::parse_record(l).unwrap())).collect()
    }

    fn store(quorum: usize) -> Store {
        Store::with_salt(Settings { quorum, page_size: 3, ..Default::default() }, [7; 16])
    }

    fn published(s: &Store) -> Vec<String> {
        let mut out = Vec::new();
        let mut cursor = 0;
        loop {
            let (c, lines, more) = s.fetch(&key(), cursor);
            out.extend(lines.iter().map(|l| l.to_string()));
            cursor = c;
            if !more {
                return out;
            }
        }
    }

    #[test]
    fn quorum_one_publishes_at_once() {
        let mut s = store(1);
        let o = s.submit(&key(), 1, 1, &recs(&[M1, T1]), 100);
        assert_eq!((o.counted, o.published), (2, 2));
        assert_eq!(published(&s), [M1, T1]);
    }

    #[test]
    fn quorum_needs_distinct_clients_and_ips() {
        let mut s = store(2);
        let r = recs(&[M1]);
        assert_eq!(s.submit(&key(), 1, 10, &r, 100).published, 0);
        // Same client again from the same IP: nothing new; from another IP: still one client.
        assert_eq!(s.submit(&key(), 1, 10, &r, 100).counted, 0);
        assert_eq!(s.submit(&key(), 1, 11, &r, 100).published, 0);
        // A second client behind the first IP: two clients but the IP is the same.
        assert_eq!(s.submit(&key(), 2, 10, &r, 100).published, 1, "client 1 also came from IP 11");
    }

    #[test]
    fn same_ip_does_not_count() {
        let mut s = store(2);
        let r = recs(&[M1]);
        s.submit(&key(), 1, 10, &r, 100);
        assert_eq!(s.submit(&key(), 2, 10, &r, 100).published, 0);
        assert!(published(&s).is_empty());
        assert_eq!(s.counts(&key()), (0, 1));
        assert_eq!(s.submit(&key(), 3, 12, &r, 100).published, 1);
        assert_eq!(published(&s), [M1]);
        assert_eq!(s.counts(&key()), (1, 0));
        // Once published, more contributions do nothing.
        assert_eq!(s.submit(&key(), 4, 13, &r, 100).counted, 0);
    }

    #[test]
    fn quorum_is_a_matching() {
        // A uses ips 1 and 2, B and C only ip 1: three clients and two ips, a matching of two.
        let mut s = store(3);
        let r = recs(&[M1]);
        s.submit(&key(), 1, 1, &r, 100);
        s.submit(&key(), 1, 2, &r, 100);
        s.submit(&key(), 2, 1, &r, 100);
        s.submit(&key(), 3, 1, &r, 100);
        assert!(published(&s).is_empty());
        s.submit(&key(), 4, 5, &r, 100);
        assert_eq!(published(&s), [M1]);
    }

    #[test]
    fn pipeline_waits_for_its_modules() {
        let mut s = store(2);
        let p = recs(&[P12]);
        // The pipeline reaches quorum first: it waits.
        s.submit(&key(), 1, 1, &p, 100);
        let o = s.submit(&key(), 2, 2, &p, 100);
        assert_eq!(o.published, 0);
        assert!(published(&s).is_empty());
        // One module published: still waiting for the other.
        s.submit(&key(), 1, 1, &recs(&[M1]), 100);
        assert_eq!(s.submit(&key(), 2, 2, &recs(&[M1]), 100).published, 1);
        assert_eq!(published(&s), [M1]);
        // The second module releases the pipeline right after it, in order.
        s.submit(&key(), 1, 1, &recs(&[M2]), 100);
        assert_eq!(s.submit(&key(), 2, 2, &recs(&[M2]), 100).published, 2);
        assert_eq!(published(&s), [M1, M2, P12]);
        assert_eq!(s.counts(&key()), (3, 0));
    }

    #[test]
    fn pipeline_before_its_modules_in_one_request() {
        let mut s = store(1);
        let o = s.submit(&key(), 1, 1, &recs(&[P12, M1, M2]), 100);
        assert_eq!(o.published, 3);
        assert_eq!(published(&s), [M1, M2, P12]);
    }

    #[test]
    fn known_records() {
        let mut s = store(1);
        let mut known = Known::default();
        known.add(M1);
        s.set_known(HashMap::from([(key(), known)]));
        // A known record is neither served nor counted.
        let o = s.submit(&key(), 1, 1, &recs(&[M1]), 100);
        assert_eq!((o.counted, o.published), (0, 0));
        // Known modules satisfy the pipelines (M2 is published, 000...1 is only known).
        let o = s.submit(&key(), 1, 1, &recs(&[M2, P12]), 100);
        assert_eq!((o.counted, o.published), (2, 2));
        assert_eq!(published(&s), [M2, P12]);
    }

    #[test]
    fn known_loaded_after_pending_cleans_and_releases() {
        let mut s = store(2);
        s.submit(&key(), 1, 1, &recs(&[M1, P12]), 100);
        s.submit(&key(), 2, 2, &recs(&[P12]), 100); // ready, waiting for modules
        assert_eq!(s.counts(&key()), (0, 2));
        let mut known = Known::default();
        known.add(M1);
        known.add(M2);
        s.set_known(HashMap::from([(key(), known)]));
        assert_eq!(published(&s), [P12]);
        assert_eq!(s.counts(&key()), (1, 0), "the pending M1 is known now");
    }

    #[test]
    fn cursor_and_more() {
        let mut s = store(1);
        let lines: Vec<String> = (1..=7).map(|i| format!("T {i:016x} x")).collect();
        s.submit(&key(), 1, 1, &strings(&lines), 100);
        let (c, got, more) = s.fetch(&key(), 0);
        assert_eq!((c, got.len(), more), (3, 3, true));
        let (c, got, more) = s.fetch(&key(), c);
        assert_eq!((c, got.len(), more), (6, 3, true));
        assert_eq!(&*got[0], lines[3]);
        let (c, got, more) = s.fetch(&key(), c);
        assert_eq!((c, got.len(), more), (7, 1, false));
        // Nothing new: the cursor stays and the list is empty.
        let (c, got, more) = s.fetch(&key(), c);
        assert_eq!((c, got.len(), more), (7, 0, false));
        // A cursor beyond the end (server restarted empty) starts over.
        let (c, got, _) = s.fetch(&key(), 99);
        assert_eq!((c, got.len()), (3, 3));
        assert_eq!(s.fetch(&PartKey::new(FMT, "PPSA00000", "1"), 5), (0, vec![], false));
    }

    #[test]
    fn partitions_are_independent() {
        let mut s = store(1);
        let other = PartKey::new(FMT, "PPSA21564", "01.008.000");
        s.submit(&key(), 1, 1, &recs(&[M1]), 100);
        s.submit(&other, 1, 1, &recs(&[M2]), 100);
        assert_eq!(published(&s), [M1]);
        assert_eq!(s.fetch(&other, 0).1.len(), 1);
    }

    #[test]
    fn limits_of_partition_budget_and_count() {
        let mut s = Store::with_salt(
            Settings { quorum: 1, partition_cap: 3, max_partitions: 1, page_size: 10 },
            [0; 16],
        );
        let lines: Vec<String> = (1..=5).map(|i| format!("T {i:016x} x")).collect();
        let r = strings(&lines);
        let o = s.submit(&key(), 1, 1, &r, 100);
        assert_eq!((o.published, o.dropped), (3, 2), "partition cap");
        let other = PartKey::new(FMT, "PPSA00001", "1");
        assert!(s.submit(&other, 1, 1, &r, 100).partition_refused);
        assert_eq!(s.partitions(), 1);

        let mut s = store(1);
        let o = s.submit(&key(), 1, 1, &r, 2);
        assert_eq!((o.counted, o.published, o.dropped), (2, 2, 3), "client budget");
    }

    #[test]
    fn persistence_roundtrip_and_export() {
        let dir = std::env::temp_dir().join(format!("pss-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut s = store(2);
        let mut known = Known::default();
        known.add(T1);
        s.set_known(HashMap::from([(key(), known)]));
        // Published: M1 (two clients). Pending: M2 (one client), P12 ready but waiting for M2.
        s.submit(&key(), 1, 1, &recs(&[M1, M2, P12, T1]), 100);
        s.submit(&key(), 2, 2, &recs(&[M1, P12]), 100);
        assert_eq!(s.counts(&key()), (1, 2));
        assert!(s.is_dirty());
        s.save(&dir).unwrap();
        assert!(!s.is_dirty());

        let export = dir.join("export").join("v1");
        let text = fs::read_to_string(export.join("PPSA21564_01.007.000.kpm")).unwrap();
        assert_eq!(text, format!("{FMT}\n{M1}\n"));
        let text = fs::read_to_string(export.join("PPSA21564_01.007.000.pending.kpm")).unwrap();
        let mut want = [M2, P12];
        want.sort_unstable();
        assert_eq!(text, format!("{FMT}\n{}\n", want.join("\n")));
        assert!(!dir.join("state.json.tmp").exists());

        let mut r = Store::load(Settings { quorum: 2, page_size: 3, ..Default::default() }, &dir).unwrap();
        assert_eq!(r.salt, s.salt);
        assert_eq!(r.hash("c", "x"), s.hash("c", "x"));
        assert_eq!(published(&r), [M1]);
        assert_eq!(r.counts(&key()), (1, 2));
        // The pending state survived: a second client completes M2 and releases P12.
        r.submit(&key(), 1, 1, &recs(&[M2]), 100);
        let o = r.submit(&key(), 3, 3, &recs(&[M2]), 100);
        assert_eq!(o.published, 2);
        assert_eq!(published(&r), [M1, M2, P12]);
        r.save(&dir).unwrap();
        let text = fs::read_to_string(export.join("PPSA21564_01.007.000.kpm")).unwrap();
        assert_eq!(text, format!("{FMT}\n{M1}\n{M2}\n{P12}\n"));
        assert!(!export.join("PPSA21564_01.007.000.pending.kpm").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_known_reads_kpm_files() {
        let dir = std::env::temp_dir().join(format!("pss-known-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("PPSA21564_01.007.000.kpm"), format!("{FMT}\r\n{M1}\r\n{P12}\r\n")).unwrap();
        fs::write(dir.join("README.txt"), "x").unwrap();
        fs::write(dir.join("bad_name.kpm"), format!("{FMT}\n{M1}\n")).unwrap();
        fs::write(dir.join("PPSA00001_1.kpm"), "no header\nM 000000000001 vs\n").unwrap();
        let known = load_known(&dir).unwrap();
        assert_eq!(known.len(), 1);
        let k = &known[&key()];
        assert_eq!(k.len(), 2);
        assert!(k.modules.contains("000000000001"));
        let _ = fs::remove_dir_all(&dir);
    }
}
