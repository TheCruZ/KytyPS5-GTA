//! In-memory state: records per partition, quorum, publication order, the copies between the
//! partitions of one game and persistence.

use crate::regions::Regions;
use crate::validate::{self, Kind, ModuleInfo, Parsed};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;
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

impl fmt::Display for PartKey {
    /// `PPSA04263_01.011.001`, as the files.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}", self.title, self.version)
    }
}

#[derive(Clone, Debug)]
pub struct Settings {
    pub quorum: usize,
    pub partition_cap: usize,
    pub max_partitions: usize,
    /// Published records reviewed per response.
    pub page_size: usize,
    /// Release whose records are left out for the clients that do not say what they have.
    pub default_release: String,
    /// Copies between the partitions of one game (see `Store::merge`); None: every partition alone.
    pub merge: Option<Merge>,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            quorum: 2,
            partition_cap: 200_000,
            max_partitions: 500,
            page_size: 20_000,
            default_release: "default".into(),
            merge: None,
        }
    }
}

/// Which partitions of one game share their records: two regions of one version always; two
/// versions when at least `share_percent` % of the shaders (`code=`) of the published modules of
/// the one with fewer are in the other, and it has at least `min_codes` of them.
#[derive(Clone, Debug)]
pub struct Merge {
    pub regions: Regions,
    pub share_percent: usize,
    pub min_codes: usize,
}

impl Default for Merge {
    fn default() -> Self {
        Self { regions: Regions::embedded(), share_percent: 50, min_codes: 100 }
    }
}

/// What one release of the repository ships for a partition (a `.kpm` of `assets/pipelines`), in
/// file order. The server publishes these records as seeds; they are not counted as new and
/// never leave the export.
#[derive(Default)]
pub struct Known {
    order: Vec<Arc<str>>,
    lines: HashSet<Arc<str>>,
}

impl Known {
    pub fn add(&mut self, line: &str) {
        if validate::parse_record(line).is_some() {
            let line: Arc<str> = Arc::from(line);
            if self.lines.insert(line.clone()) {
                self.order.push(line);
            }
        }
    }

    pub fn len(&self) -> usize {
        self.lines.len()
    }

    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }
}

/// The records of one release: the `.kpm` files of a folder of `--known`.
pub struct Release {
    pub name: String,
    pub parts: HashMap<PartKey, Known>,
}

/// The release the loose `.kpm` files directly in the `--known` folder belong to.
pub const LOOSE_RELEASE: &str = "default";

/// Which published records a response leaves out.
#[derive(Debug, Default)]
pub enum Filter {
    /// Nothing is left out.
    #[default]
    None,
    /// What the default release ships for the partition (clients that do not say what they have).
    DefaultRelease,
    /// What the client has: hashes of records (`record_hash`).
    Have(HashSet<u64>),
}

/// First 8 bytes of the SHA-256 of a record line: what the clients call `RecordHash`.
pub fn record_hash(line: &str) -> u64 {
    let d = Sha256::digest(line.as_bytes());
    u64::from_be_bytes(d[..8].try_into().unwrap_or_default())
}

/// Parses the 16 hex digits of a record hash.
pub fn parse_hash(s: &str) -> Option<u64> {
    if s.len() != 16 || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    u64::from_str_radix(s, 16).ok()
}

fn is_known(releases: &[Release], key: &PartKey, line: &str) -> bool {
    releases.iter().any(|r| r.parts.get(key).is_some_and(|k| k.lines.contains(line)))
}

struct Pending {
    pairs: Vec<(u64, u64)>,
    /// Reached quorum but waits for the modules of a pipeline.
    ready: bool,
}

enum Entry {
    /// Published as its own: its players sent it (with quorum) or its release ships it.
    Published,
    Pending(Pending),
    /// Published as a copy of the own record of a linked partition (`Store::merge`): served like
    /// the others, never exported. The pairs are the clients of this partition that sent it; with
    /// quorum it becomes its own.
    Copy(Vec<(u64, u64)>),
}

/// In `published`, what takes the place of a record that is no longer served (a pipeline no draw
/// creates from its modules): the sequence numbers of the records after it, the cursors of the
/// clients, stay. Saved as an empty line.
fn is_tombstone(line: &str) -> bool {
    line.is_empty()
}

/// The shader of a module record (`code=`, 16 hex digits).
fn module_code(line: &str) -> Option<u64> {
    let value = line.strip_prefix("M ")?.split(' ').find_map(|t| t.strip_prefix("code="))?;
    Some(u64::from_str_radix(value, 16).unwrap_or_else(|_| record_hash(value)))
}

/// Adds a (client, ip) pair to those of a record; false when it adds nothing.
fn add_pair(pairs: &mut Vec<(u64, u64)>, client: u64, ip: u64) -> bool {
    let by_client = pairs.iter().filter(|q| q.0 == client).count();
    if pairs.contains(&(client, ip)) || pairs.len() >= MAX_PAIRS || by_client >= MAX_PAIRS_PER_CLIENT {
        return false;
    }
    pairs.push((client, ip));
    true
}

struct Module {
    info: ModuleInfo,
    line: Arc<str>,
}

#[derive(Default)]
struct Partition {
    published: Vec<Arc<str>>,
    index: HashMap<Arc<str>, Entry>,
    /// The published modules by id (the first record of an id, as the emulator reads them).
    modules: HashMap<String, Module>,
    waiting: Vec<Arc<str>>,
    dirty: bool,
    /// Records in `index` that are copies.
    copies: usize,
    /// Copies added since the start (for the log).
    copied: usize,
    /// The shaders of its own published modules: what decides whether two versions are close.
    codes: HashSet<u64>,
    /// How many shaders it had when its links were last reviewed (None: never).
    codes_reviewed: Option<usize>,
    /// Records that became its own since they were last copied to the linked partitions.
    fresh: Vec<Arc<str>>,
}

/// Records of a partition left out because no draw creates their pipeline from the modules
/// they name: how many, and why the first one was.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Unlinked {
    pub count: usize,
    pub example: Option<String>,
}

impl Unlinked {
    fn add(&mut self, line: &str, why: String) {
        self.count += 1;
        if self.example.is_none() {
            self.example = Some(format!("{line} ({why})"));
        }
    }
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
    fn deps_ok(&self, parsed: &Parsed) -> bool {
        parsed.deps.iter().all(|d| self.modules.contains_key(d))
    }

    /// Why no draw creates the pipeline of a record whose modules are all published (see
    /// `validate::pipeline_link_error`), or None (also for the other records).
    fn link_error(&self, line: &str, parsed: &Parsed) -> Option<String> {
        if parsed.kind != Kind::Pipeline {
            return None;
        }
        validate::pipeline_link_error(line, |id| self.modules.get(id).map(|m| &m.info))
    }

    fn add_module(&mut self, line: &Arc<str>, parsed: &Parsed) {
        if let Some(id) = &parsed.id {
            if let Some(info) = validate::module_info(line) {
                self.modules.entry(id.clone()).or_insert(Module { info, line: line.clone() });
            }
        }
    }

    /// Assigns the next sequence number. Returns whether it was a module.
    fn publish(&mut self, line: &Arc<str>, parsed: &Parsed) -> bool {
        self.index.insert(line.clone(), Entry::Published);
        self.published.push(line.clone());
        self.fresh.push(line.clone());
        self.dirty = true;
        self.add_module(line, parsed);
        if parsed.id.is_some() {
            self.codes.extend(module_code(line));
            return true;
        }
        self.own_modules_of(parsed);
        false
    }

    /// The copied modules that one of its own pipelines names are its own too: its players
    /// recorded them (the id of a module is the hash of its line) but did not send them, as they
    /// had them already, and its export needs them.
    fn own_modules_of(&mut self, parsed: &Parsed) {
        for dep in &parsed.deps {
            if let Some(line) = self.modules.get(dep).map(|m| m.line.clone()) {
                self.promote(&line);
            }
        }
    }

    /// Makes a copy its own record, in its place. Returns whether it was a copy.
    fn promote(&mut self, line: &Arc<str>) -> bool {
        let Some(entry) = self.index.get_mut(&**line) else { return false };
        if !matches!(entry, Entry::Copy(_)) {
            return false;
        }
        *entry = Entry::Published;
        self.copies -= 1;
        self.codes.extend(module_code(line));
        self.fresh.push(line.clone());
        self.dirty = true;
        if let Some(parsed) = validate::parse_record(line) {
            self.own_modules_of(&parsed);
        }
        true
    }

    /// Publishes the own record of a linked partition as a copy, if this one has no record of
    /// its line (or, for a module, of its id) and, for a pipeline, has its modules and a draw
    /// creates it from them. A pending record keeps its clients; one that has its quorum and only
    /// waits for its modules is left to `resolve_waiting`, as its own. Tessellation records are not
    /// copied: the emulator does not take them from the shared manifest. Returns whether it copied.
    fn add_copy(&mut self, line: &Arc<str>, parsed: &Parsed, cap: usize) -> bool {
        match parsed.kind {
            Kind::Tessellation => return false,
            Kind::Module => {
                if parsed.id.as_ref().is_some_and(|id| self.modules.contains_key(id)) {
                    return false;
                }
            }
            Kind::Pipeline => {
                if !self.deps_ok(parsed) || self.link_error(line, parsed).is_some() {
                    return false;
                }
            }
        }
        let (line, pairs) = match self.index.get_key_value(&**line) {
            Some((k, Entry::Pending(p))) if !p.ready => (k.clone(), p.pairs.clone()),
            Some(_) => return false,
            None if self.copies >= cap => return false,
            None => (line.clone(), Vec::new()),
        };
        self.index.insert(line.clone(), Entry::Copy(pairs));
        self.published.push(line.clone());
        self.copies += 1;
        self.copied += 1;
        self.dirty = true;
        self.add_module(&line, parsed);
        true
    }

    /// Publishes the pipelines that were waiting for modules and now have them; those that no
    /// draw creates from them are dropped and counted in `unlinked`.
    fn resolve_waiting(&mut self, unlinked: &mut Unlinked) -> usize {
        let mut published = 0;
        let waiting = std::mem::take(&mut self.waiting);
        for line in waiting {
            if matches!(self.index.get(&*line), Some(Entry::Published | Entry::Copy(_))) {
                continue;
            }
            match validate::parse_record(&line) {
                Some(parsed) if self.deps_ok(&parsed) => {
                    if let Some(why) = self.link_error(&line, &parsed) {
                        self.index.remove(&*line);
                        self.dirty = true;
                        unlinked.add(&line, why);
                        continue;
                    }
                    self.publish(&line, &parsed);
                    published += 1;
                }
                _ => self.waiting.push(line),
            }
        }
        published
    }

    /// Takes the pipelines that no draw creates from their modules out of what is published (a
    /// tombstone keeps their place) and out of what waits. Returns (published, pending) removed.
    fn drop_unlinkable(&mut self, unlinked: &mut Unlinked) -> (usize, usize) {
        let mut published = 0;
        for i in 0..self.published.len() {
            let line = self.published[i].clone();
            let Some(parsed) = validate::parse_record(&line) else { continue };
            if !self.deps_ok(&parsed) {
                continue;
            }
            if let Some(why) = self.link_error(&line, &parsed) {
                if matches!(self.index.remove(&*line), Some(Entry::Copy(_))) {
                    self.copies -= 1;
                }
                self.published[i] = Arc::from("");
                unlinked.add(&line, why);
                published += 1;
            }
        }
        let mut pending: Vec<Arc<str>> = Vec::new();
        for (line, entry) in &self.index {
            if !matches!(entry, Entry::Pending(_)) {
                continue;
            }
            let Some(parsed) = validate::parse_record(line) else { continue };
            if self.deps_ok(&parsed) {
                if let Some(why) = self.link_error(line, &parsed) {
                    unlinked.add(line, why);
                    pending.push(line.clone());
                }
            }
        }
        for line in &pending {
            self.index.remove(&**line);
        }
        self.waiting.retain(|l| !pending.contains(l));
        if published + pending.len() > 0 {
            self.dirty = true;
        }
        (published, pending.len())
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct SubmitOutcome {
    /// New (client, record) contributions, for the daily limits.
    pub counted: u64,
    pub published: usize,
    /// Dropped because the partition is full or the daily budget ran out.
    pub dropped: usize,
    /// Pipelines left out because no draw creates them from the modules they name.
    pub unlinked: Unlinked,
    /// The partition does not exist and no more may be created.
    pub partition_refused: bool,
    /// Copies published in the linked partitions.
    pub copied: usize,
}

/// Two partitions of one game, as the startup log shows them.
#[derive(Debug, PartialEq, Eq)]
pub struct PairReport {
    pub a: PartKey,
    pub b: PartKey,
    /// Shaders of the one with fewer that the other has, and how many it has.
    pub common: usize,
    pub fewer: usize,
    pub linked: bool,
}

/// The partitions of one game (and format) after `Store::merge`.
#[derive(Debug, PartialEq, Eq)]
pub struct MergeReport {
    pub game: String,
    pub format: String,
    pub pairs: Vec<PairReport>,
    /// Copies its partitions hold, and how many of them were added since the start.
    pub copies: usize,
    pub added: usize,
}

impl fmt::Display for PairReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let percent = if self.fewer == 0 { 0.0 } else { 100.0 * self.common as f64 / self.fewer as f64 };
        write!(f, "{} + {} {percent:.1}% ({} of {})", self.a, self.b, self.common, self.fewer)
    }
}

impl fmt::Display for MergeReport {
    /// `merge GTA V (v1): together A + B 91.6% (1283 of 1400), ...; apart C + D 4.2% (23 of 548);
    /// copies 8123 (8123 new)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let list = |linked: bool| {
            let pairs: Vec<String> = self.pairs.iter().filter(|p| p.linked == linked).map(|p| p.to_string()).collect();
            if pairs.is_empty() { "none".to_string() } else { pairs.join(", ") }
        };
        write!(
            f,
            "merge {} ({}): together {}; apart {}; copies {} ({} new)",
            self.game,
            validate::format_dir(&self.format),
            list(true),
            list(false),
            self.copies,
            self.added
        )
    }
}

pub struct Store {
    settings: Settings,
    salt: [u8; 16],
    parts: HashMap<PartKey, Partition>,
    releases: Vec<Release>,
    state_dirty: bool,
    /// Pairs of partitions of one game that share their records (the first one is the lesser key).
    /// A pair, once linked, stays linked: after the copies its players no longer send what they
    /// received, and the shaders the two have in common would seem fewer than they are.
    links: BTreeSet<(PartKey, PartKey)>,
    /// The release the home page compares with (the latest one); never published.
    reference: Option<Release>,
}

/// What the home page shows of a partition.
#[derive(Debug, PartialEq, Eq)]
pub struct PartStats {
    pub key: PartKey,
    /// Distinct shaders (`code=`) of its own published modules, and how many of them the
    /// reference release ships.
    pub shaders: usize,
    pub shaders_in_release: usize,
    /// Its own published pipelines, and how many of them the reference release ships.
    pub pipelines: usize,
    pub pipelines_in_release: usize,
    /// What it holds as copies of other regions and versions: shaders that none of its own
    /// modules has, and pipelines.
    pub copied_shaders: usize,
    pub copied_pipelines: usize,
}

impl Store {
    pub fn new(settings: Settings) -> Self {
        Self::with_salt(settings, rand::random())
    }

    pub fn with_salt(settings: Settings, salt: [u8; 16]) -> Self {
        Self {
            settings,
            salt,
            parts: HashMap::new(),
            releases: Vec::new(),
            state_dirty: false,
            links: BTreeSet::new(),
            reference: None,
        }
    }

    /// Sets the release the home page compares with.
    pub fn set_reference(&mut self, release: Option<Release>) {
        self.reference = release;
    }

    pub fn reference_name(&self) -> Option<&str> {
        self.reference.as_ref().map(|r| r.name.as_str())
    }

    /// Per partition that has anything: its own shaders and pipelines, how many of them the
    /// reference release ships (the rest came from the players, or reached the repository after
    /// the release) and what it holds as copies. In key order.
    pub fn stats(&self) -> Vec<PartStats> {
        let mut out = Vec::new();
        for (key, part) in &self.parts {
            let release = self.reference.as_ref().and_then(|r| r.parts.get(key));
            let release_codes: HashSet<u64> =
                release.map(|k| k.order.iter().filter_map(|l| module_code(l)).collect()).unwrap_or_default();
            let mut s = PartStats {
                key: key.clone(),
                shaders: part.codes.len(),
                shaders_in_release: part.codes.iter().filter(|c| release_codes.contains(c)).count(),
                pipelines: 0,
                pipelines_in_release: 0,
                copied_shaders: 0,
                copied_pipelines: 0,
            };
            let mut copied_codes = HashSet::new();
            for line in &part.published {
                let pipeline = line.starts_with("P ");
                match part.index.get(&**line) {
                    Some(Entry::Published) if pipeline => {
                        s.pipelines += 1;
                        s.pipelines_in_release += usize::from(release.is_some_and(|k| k.lines.contains(&**line)));
                    }
                    Some(Entry::Copy(_)) if pipeline => s.copied_pipelines += 1,
                    Some(Entry::Copy(_)) => {
                        copied_codes.extend(module_code(line).filter(|c| !part.codes.contains(c)));
                    }
                    _ => {}
                }
            }
            s.copied_shaders = copied_codes.len();
            if s.shaders + s.pipelines + s.copied_shaders + s.copied_pipelines > 0 {
                out.push(s);
            }
        }
        out.sort_by(|a, b| a.key.cmp(&b.key));
        out
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    pub fn settings_mut(&mut self) -> &mut Settings {
        &mut self.settings
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

    /// (published, pending) records of a partition; the published ones include its copies.
    pub fn counts(&self, key: &PartKey) -> (usize, usize) {
        self.parts.get(key).map_or((0, 0), |p| {
            let published = p.published.iter().filter(|l| !is_tombstone(l)).count();
            let pending = p.index.values().filter(|e| matches!(e, Entry::Pending(_))).count();
            (published, pending)
        })
    }

    /// Copies a partition holds.
    pub fn copies(&self, key: &PartKey) -> usize {
        self.parts.get(key).map_or(0, |p| p.copies)
    }

    /// Replaces the releases of the repository and publishes the records of each one that are not
    /// published yet, in file order and release order. Pending records that a release ships are
    /// published too. A pipeline waits until its modules are published (the modules of every
    /// release count). Pipelines that no draw creates from the modules they name are left out:
    /// returns them by partition.
    pub fn set_known(&mut self, releases: Vec<Release>) -> Vec<(PartKey, Unlinked)> {
        let mut rejected = Vec::new();
        self.releases = releases;
        let mut keys: HashSet<PartKey> = self.parts.keys().cloned().collect();
        for r in &self.releases {
            keys.extend(r.parts.keys().cloned());
        }
        for key in keys {
            let in_release = self.releases.iter().any(|r| r.parts.contains_key(&key));
            let part = self.parts.entry(key.clone()).or_default();
            let mut published = 0;
            let mut unlinked = Unlinked::default();
            if in_release {
                part.waiting.retain(|l| !is_known(&self.releases, &key, l));
                for release in &self.releases {
                    let Some(known) = release.parts.get(&key) else { continue };
                    for line in &known.order {
                        match part.index.get(&**line) {
                            Some(Entry::Published) => continue,
                            // What its release ships is its own.
                            Some(Entry::Copy(_)) => {
                                part.promote(line);
                                continue;
                            }
                            _ => {}
                        }
                        let Some(parsed) = validate::parse_record(line) else { continue };
                        if !part.deps_ok(&parsed) {
                            part.waiting.push(line.clone());
                        } else if let Some(why) = part.link_error(line, &parsed) {
                            unlinked.add(line, why);
                        } else {
                            part.publish(line, &parsed);
                            published += 1;
                        }
                    }
                    published += part.resolve_waiting(&mut unlinked);
                }
            }
            published += part.resolve_waiting(&mut unlinked);
            self.state_dirty |= published > 0 || part.dirty;
            if unlinked.count > 0 {
                rejected.push((key, unlinked));
            }
        }
        rejected.sort_by(|a, b| a.0.cmp(&b.0));
        rejected
    }

    /// Takes out of every partition the pipelines that no draw creates from the modules they
    /// name (a state saved before the server checked them): those published keep their place as a
    /// tombstone, which is neither served nor exported. Returns, by partition, the published and
    /// pending records removed.
    pub fn drop_unlinkable(&mut self) -> Vec<(PartKey, usize, usize, Unlinked)> {
        let mut out = Vec::new();
        for (key, part) in &mut self.parts {
            let mut unlinked = Unlinked::default();
            let (published, pending) = part.drop_unlinkable(&mut unlinked);
            if published + pending > 0 {
                self.state_dirty = true;
                out.push((key.clone(), published, pending, unlinked));
            }
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
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
        let room = self.parts.contains_key(key) || self.parts.len() < self.settings.max_partitions;
        if records.is_empty() || budget == 0 {
            out.dropped = records.len();
            // The first client of a region whose version another region has gets its records,
            // even before it sends any.
            if room && !self.parts.contains_key(key) && self.has_same_version(key) {
                self.parts.insert(key.clone(), Partition::default());
                out.copied = self.spread_fresh();
            }
            return out;
        }
        if !room {
            out.partition_refused = true;
            return out;
        }
        let releases = &self.releases;
        let part = self.parts.entry(key.clone()).or_default();
        let quorum = self.settings.quorum;
        let cap = self.settings.partition_cap;
        let mut new_modules = false;
        for (line, parsed) in records {
            if is_known(releases, key, line) {
                continue;
            }
            // A pipeline whose modules are all published is checked at once; one that waits for
            // them, when they are.
            if part.deps_ok(parsed) {
                if let Some(why) = part.link_error(line, parsed) {
                    if matches!(part.index.get(line.as_str()), Some(Entry::Pending(_))) {
                        part.index.remove(line.as_str());
                        part.waiting.retain(|l| **l != **line);
                        part.dirty = true;
                    }
                    out.unlinked.add(line, why);
                    continue;
                }
            }
            if budget == 0 {
                out.dropped += 1;
                continue;
            }
            let line_arc: Arc<str>;
            if let Some(k) = part.index.get_key_value(line.as_str()).map(|(k, _)| k.clone()) {
                line_arc = k;
                match part.index.get_mut(line.as_str()) {
                    Some(Entry::Pending(p)) => {
                        if !add_pair(&mut p.pairs, client, ip) {
                            continue;
                        }
                        budget -= 1;
                        out.counted += 1;
                        if p.ready || !quorum_met(&p.pairs, quorum) {
                            continue;
                        }
                        p.ready = true;
                    }
                    // A copy its own players send: with quorum, it becomes its own.
                    Some(Entry::Copy(pairs)) => {
                        if !add_pair(pairs, client, ip) {
                            continue;
                        }
                        budget -= 1;
                        out.counted += 1;
                        if quorum_met(pairs, quorum) {
                            part.promote(&line_arc);
                        }
                        continue;
                    }
                    _ => continue,
                }
            } else {
                // Copies have their own limit (`add_copy`).
                if part.index.len() - part.copies >= cap {
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
            if part.deps_ok(parsed) {
                new_modules |= part.publish(&line_arc, parsed);
                out.published += 1;
            } else {
                part.waiting.push(line_arc);
            }
        }
        if new_modules {
            out.published += part.resolve_waiting(&mut out.unlinked);
        }
        if out.counted > 0 || part.dirty {
            part.dirty = true;
            self.state_dirty = true;
        }
        out.copied = self.spread_fresh();
        out
    }

    // ---- Copies between the partitions of one game ----

    /// Whether two partitions are of one game: same format, another title of its line of the
    /// regions table or another version of its title.
    fn same_game(merge: &Merge, a: &PartKey, b: &PartKey) -> bool {
        a != b && a.format == b.format && merge.regions.same_game(&a.title, &b.title)
    }

    /// Whether another region has a partition of the same version.
    fn has_same_version(&self, key: &PartKey) -> bool {
        let Some(merge) = &self.settings.merge else { return false };
        self.parts.keys().any(|k| k.version == key.version && Self::same_game(merge, k, key))
    }

    /// Shaders of the partition with fewer that the other has, and how many it has.
    fn share(&self, a: &PartKey, b: &PartKey) -> (usize, usize) {
        let (Some(a), Some(b)) = (self.parts.get(a), self.parts.get(b)) else { return (0, 0) };
        let (fewer, more) = if a.codes.len() <= b.codes.len() { (&a.codes, &b.codes) } else { (&b.codes, &a.codes) };
        (fewer.iter().filter(|c| more.contains(c)).count(), fewer.len())
    }

    fn should_link(&self, merge: &Merge, a: &PartKey, b: &PartKey) -> bool {
        if a.version == b.version {
            return true;
        }
        let (common, fewer) = self.share(a, b);
        fewer >= merge.min_codes.max(1) && common * 100 >= merge.share_percent * fewer
    }

    fn pair(a: &PartKey, b: &PartKey) -> (PartKey, PartKey) {
        if a <= b { (a.clone(), b.clone()) } else { (b.clone(), a.clone()) }
    }

    /// The partitions linked with `key`, in key order.
    fn linked(&self, key: &PartKey) -> Vec<PartKey> {
        let mut out: Vec<PartKey> = self
            .links
            .iter()
            .filter_map(|(a, b)| if a == key { Some(b.clone()) } else if b == key { Some(a.clone()) } else { None })
            .collect();
        out.sort();
        out
    }

    /// Reviews the pairs of `key` that are not linked yet and links those that should be. Returns
    /// the partitions it was linked with now.
    fn relink(&mut self, key: &PartKey) -> Vec<PartKey> {
        let Some(merge) = self.settings.merge.clone() else { return Vec::new() };
        let Some(part) = self.parts.get_mut(key) else { return Vec::new() };
        part.codes_reviewed = Some(part.codes.len());
        let mut others: Vec<PartKey> = self
            .parts
            .keys()
            .filter(|k| Self::same_game(&merge, key, k) && !self.links.contains(&Self::pair(key, k)))
            .filter(|k| self.should_link(&merge, key, k))
            .cloned()
            .collect();
        others.sort();
        for other in &others {
            self.links.insert(Self::pair(key, other));
            self.state_dirty = true;
        }
        others
    }

    /// Copies to `to` the own records among `lines` of `from` that it lacks, in their order; a
    /// pipeline goes after the modules it names, which go with it whether they are its own or
    /// copies in `from`. Pipelines that `to` cannot link are left out. Then `to` publishes the
    /// pipelines of its own that waited for those modules. Returns the copies.
    fn copy_lines(&mut self, from: &PartKey, lines: &[Arc<str>], to: &PartKey) -> usize {
        let cap = self.settings.partition_cap;
        let Some(src) = self.parts.get(from) else { return 0 };
        let mut todo: Vec<(Arc<str>, Parsed)> = Vec::new();
        let mut queued: HashSet<Arc<str>> = HashSet::new();
        for line in lines {
            if !matches!(src.index.get(&**line), Some(Entry::Published)) {
                continue;
            }
            let Some(parsed) = validate::parse_record(line) else { continue };
            if parsed.kind == Kind::Tessellation {
                continue;
            }
            for dep in &parsed.deps {
                let Some(module) = src.modules.get(dep) else { continue };
                if queued.insert(module.line.clone()) {
                    if let Some(p) = validate::parse_record(&module.line) {
                        todo.push((module.line.clone(), p));
                    }
                }
            }
            if queued.insert(line.clone()) {
                todo.push((line.clone(), parsed));
            }
        }
        let Some(dst) = self.parts.get_mut(to) else { return 0 };
        let mut copied = 0;
        for (line, parsed) in &todo {
            copied += usize::from(dst.add_copy(line, parsed, cap));
        }
        if copied > 0 {
            dst.resolve_waiting(&mut Unlinked::default());
            self.state_dirty = true;
        }
        copied
    }

    /// Copies all the own records of `from` that `to` lacks.
    fn backfill(&mut self, from: &PartKey, to: &PartKey) -> usize {
        let lines = self.parts.get(from).map(|p| p.published.clone()).unwrap_or_default();
        self.copy_lines(from, &lines, to)
    }

    /// Reviews the links of the partitions whose shaders grew (or never reviewed) and copies,
    /// between the pairs that were just linked, everything the other lacks; then copies the
    /// records that became own since the last time to the linked partitions, until nothing
    /// changes. Returns the copies.
    fn spread_fresh(&mut self) -> usize {
        if self.settings.merge.is_none() {
            for part in self.parts.values_mut() {
                part.fresh.clear();
            }
            return 0;
        }
        let mut copied = 0;
        loop {
            let mut keys: Vec<PartKey> = self
                .parts
                .iter()
                .filter(|(_, p)| !p.fresh.is_empty() || p.codes_reviewed != Some(p.codes.len()))
                .map(|(k, _)| k.clone())
                .collect();
            if keys.is_empty() {
                return copied;
            }
            keys.sort();
            for key in keys {
                let Some(part) = self.parts.get(&key) else { continue };
                if part.codes_reviewed != Some(part.codes.len()) {
                    for other in self.relink(&key) {
                        copied += self.backfill(&key, &other);
                        copied += self.backfill(&other, &key);
                    }
                }
                let Some(part) = self.parts.get_mut(&key) else { continue };
                let fresh = std::mem::take(&mut part.fresh);
                if fresh.is_empty() {
                    continue;
                }
                for other in self.linked(&key) {
                    copied += self.copy_lines(&key, &fresh, &other);
                }
            }
        }
    }

    /// At start, after the seeds: links the partitions of one game that should be (a pair linked
    /// before stays so, unless the regions table no longer puts them in one game) and copies
    /// between every linked pair what the other lacks, in key order, so that the result does not
    /// depend on the order the records arrived in. Returns one report per game (and format) with
    /// more than one partition.
    pub fn merge(&mut self) -> Vec<MergeReport> {
        for part in self.parts.values_mut() {
            part.fresh.clear();
        }
        let Some(merge) = self.settings.merge.clone() else { return Vec::new() };
        let before = self.links.len();
        self.links.retain(|(a, b)| Self::same_game(&merge, a, b));
        self.state_dirty |= self.links.len() != before;
        let added_before: HashMap<PartKey, usize> = self.parts.iter().map(|(k, p)| (k.clone(), p.copied)).collect();
        let mut keys: Vec<PartKey> = self.parts.keys().cloned().collect();
        keys.sort();
        for key in &keys {
            self.relink(key);
        }
        for (a, b) in self.links.clone() {
            self.backfill(&a, &b);
            self.backfill(&b, &a);
        }
        self.spread_fresh();

        let mut games: Vec<(String, String, Vec<PartKey>)> = Vec::new();
        for key in &keys {
            let game = merge.regions.game_name(&key.title).to_string();
            match games.iter_mut().find(|g| g.0 == game && g.1 == key.format) {
                Some(g) => g.2.push(key.clone()),
                None => games.push((game, key.format.clone(), vec![key.clone()])),
            }
        }
        let mut reports = Vec::new();
        for (game, format, members) in games {
            if members.len() < 2 {
                continue;
            }
            let mut pairs = Vec::new();
            for (i, a) in members.iter().enumerate() {
                for b in &members[i + 1..] {
                    let (common, fewer) = self.share(a, b);
                    let linked = self.links.contains(&Self::pair(a, b));
                    pairs.push(PairReport { a: a.clone(), b: b.clone(), common, fewer, linked });
                }
            }
            let copies = members.iter().map(|k| self.parts[k].copies).sum();
            let added = members.iter().map(|k| self.parts[k].copied - added_before.get(k).copied().unwrap_or(0)).sum();
            reports.push(MergeReport { game, format, pairs, copies, added });
        }
        reports
    }

    /// Published records after `cursor`: (new cursor, records, more). A cursor beyond the
    /// published records (the server lost state) restarts from the beginning. At most `page_size`
    /// published records are reviewed; the cursor stops at the last one reviewed, so a page whose
    /// records are all left out by `filter` comes back empty with `more` set.
    pub fn fetch(&self, key: &PartKey, cursor: u64, filter: &Filter) -> (u64, Vec<Arc<str>>, bool) {
        let Some(part) = self.parts.get(key) else {
            return (0, Vec::new(), false);
        };
        let len = part.published.len();
        let from = usize::try_from(cursor).ok().filter(|c| *c <= len).unwrap_or(0);
        let to = len.min(from + self.settings.page_size);
        let page = &part.published[from..to];
        let page: Vec<Arc<str>> = page.iter().filter(|l| !is_tombstone(l)).cloned().collect();
        let page = page.as_slice();
        let lines = match filter {
            Filter::None => page.to_vec(),
            Filter::DefaultRelease => {
                let default = self
                    .releases
                    .iter()
                    .find(|r| r.name == self.settings.default_release)
                    .and_then(|r| r.parts.get(key));
                match default {
                    Some(k) => page.iter().filter(|l| !k.lines.contains(&***l)).cloned().collect(),
                    None => page.to_vec(),
                }
            }
            Filter::Have(have) => page.iter().filter(|l| !have.contains(&record_hash(l))).cloned().collect(),
        };
        (to as u64, lines, to < len)
    }

    /// Whether a release with this name is loaded.
    pub fn has_release(&self, name: &str) -> bool {
        self.releases.iter().any(|r| r.name == name)
    }

    pub fn release_names(&self) -> Vec<&str> {
        self.releases.iter().map(|r| r.name.as_str()).collect()
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
                            Entry::Published | Entry::Copy(_) => None,
                        })
                        .collect();
                    pending.sort_by(|a, b| a.line.cmp(b.line));
                    // A copy is saved as a tombstone in its place and listed apart: a server
                    // without copies serves and exports none of them, and its cursors still hold.
                    let mut copies = Vec::new();
                    let published: Vec<&str> = part
                        .published
                        .iter()
                        .enumerate()
                        .map(|(at, line)| match part.index.get(&**line) {
                            Some(Entry::Copy(pairs)) => {
                                copies.push(CopyView { at, line, pairs });
                                ""
                            }
                            _ => &**line,
                        })
                        .collect();
                    PartView {
                        format: &key.format,
                        title: &key.title,
                        version: &key.version,
                        published,
                        pending,
                        copies,
                    }
                })
                .collect();
            let links: Vec<[&str; 5]> =
                self.links.iter().map(|(a, b)| [&*a.format, &*a.title, &*a.version, &*b.title, &*b.version]).collect();
            let view = StateView { version: STATE_VERSION, salt: hex(&self.salt), partitions, links };
            write_atomic(&dir.join("state.json"), |w| {
                serde_json::to_writer(&mut *w, &view).map_err(io::Error::other)
            })?;
        }
        for key in &keys {
            let part = &self.parts[key];
            if !part.dirty {
                continue;
            }
            let folder = dir.join("export").join(validate::format_dir(&key.format));
            fs::create_dir_all(&folder)?;
            let base = key.to_string();
            // Its own records only: copies belong to the export of the partition they come from.
            let published: Vec<&str> = part
                .published
                .iter()
                .filter(|l| matches!(part.index.get(&***l), Some(Entry::Published)))
                .map(|l| &**l)
                .filter(|l| !is_known(&self.releases, key, l))
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
                if !is_tombstone(&line) {
                    part.index.insert(line.clone(), Entry::Published);
                    part.codes.extend(module_code(&line));
                }
                part.published.push(line);
            }
            // Copies go back to their places, which the list keeps as tombstones (a server
            // without copies may have published one of them since as its own: that one stays).
            for c in p.copies {
                let free = part.published.get(c.at).is_some_and(|l| is_tombstone(l));
                if free && !part.index.contains_key(c.line.as_str()) && validate::parse_record(&c.line).is_some() {
                    let line: Arc<str> = Arc::from(c.line);
                    part.index.insert(line.clone(), Entry::Copy(c.pairs));
                    part.published[c.at] = line;
                    part.copies += 1;
                }
            }
            // The modules in publication order: the first record of an id is the one used.
            for i in 0..part.published.len() {
                let line = part.published[i].clone();
                if let Some(parsed) = validate::parse_record(&line) {
                    part.add_module(&line, &parsed);
                }
            }
            for q in p.pending {
                let line: Arc<str> = Arc::from(q.line);
                if part.index.contains_key(&line) {
                    continue;
                }
                if q.ready {
                    part.waiting.push(line.clone());
                }
                part.index.insert(line, Entry::Pending(Pending { pairs: q.pairs, ready: q.ready }));
            }
            store.parts.insert(key, part);
        }
        for [format, title_a, version_a, title_b, version_b] in state.links {
            let a = PartKey { format: format.clone(), title: title_a, version: version_a };
            let b = PartKey { format, title: title_b, version: version_b };
            if a != b && store.parts.contains_key(&a) && store.parts.contains_key(&b) {
                store.links.insert(Self::pair(&a, &b));
            }
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
    /// Linked pairs: format, title and version of the first, title and version of the second.
    links: Vec<[&'a str; 5]>,
}

#[derive(Serialize)]
struct PartView<'a> {
    format: &'a str,
    title: &'a str,
    version: &'a str,
    published: Vec<&'a str>,
    pending: Vec<PendingView<'a>>,
    copies: Vec<CopyView<'a>>,
}

#[derive(Serialize)]
struct CopyView<'a> {
    at: usize,
    line: &'a str,
    #[serde(skip_serializing_if = "<[_]>::is_empty")]
    pairs: &'a [(u64, u64)],
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
    /// Absent in the state of a server without copies.
    #[serde(default)]
    links: Vec<[String; 5]>,
}

#[derive(Deserialize)]
struct PartFile {
    format: String,
    title: String,
    version: String,
    published: Vec<String>,
    pending: Vec<PendingFile>,
    #[serde(default)]
    copies: Vec<CopyFile>,
}

#[derive(Deserialize)]
struct CopyFile {
    at: usize,
    line: String,
    #[serde(default)]
    pairs: Vec<(u64, u64)>,
}

#[derive(Deserialize)]
struct PendingFile {
    line: String,
    pairs: Vec<(u64, u64)>,
    ready: bool,
}

/// Loads the `.kpm` files of one folder (`<TITLE>_<VERSION>.kpm`, as `assets/pipelines`).
pub fn load_release_dir(dir: &Path) -> io::Result<HashMap<PartKey, Known>> {
    let mut all: HashMap<PartKey, Known> = HashMap::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
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

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Chunk {
    Number(u64),
    Text(String),
}

/// "v9" before "v10": digit runs compare as numbers.
pub(crate) fn natural_key(name: &str) -> Vec<Chunk> {
    let mut out = Vec::new();
    let mut rest = name;
    while let Some(c) = rest.chars().next() {
        let digits = c.is_ascii_digit();
        let end = rest.find(|x: char| x.is_ascii_digit() != digits).unwrap_or(rest.len());
        let (head, tail) = rest.split_at(end);
        out.push(if digits {
            Chunk::Number(head.parse().unwrap_or(u64::MAX))
        } else {
            Chunk::Text(head.to_string())
        });
        rest = tail;
    }
    out
}

/// Loads the releases of a `--known` folder: every subfolder is a release (`known/v19/*.kpm`),
/// in natural name order; the `.kpm` files directly in the folder are the release `default`
/// (the older layout), loaded first.
pub fn load_known(dir: &Path) -> io::Result<Vec<Release>> {
    let mut releases = Vec::new();
    let loose = load_release_dir(dir)?;
    if !loose.is_empty() {
        releases.push(Release { name: LOOSE_RELEASE.to_string(), parts: loose });
    }
    let mut folders: Vec<(String, PathBuf)> = Vec::new();
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()).map(str::to_string) else { continue };
        let name_ok = !name.is_empty()
            && !name.starts_with('.')
            && name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'));
        if path.is_dir() && name_ok {
            folders.push((name, path));
        }
    }
    folders.sort_by_key(|(name, _)| natural_key(name));
    for (name, path) in folders {
        if releases.iter().any(|r: &Release| r.name == name) {
            return Err(io::Error::other(format!("release {name} is defined twice")));
        }
        releases.push(Release { name, parts: load_release_dir(&path)? });
    }
    Ok(releases)
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

    fn release(name: &str, known: Known) -> Release {
        Release { name: name.into(), parts: HashMap::from([(key(), known)]) }
    }

    fn known_of(lines: &[&str]) -> Known {
        let mut k = Known::default();
        for l in lines {
            k.add(l);
        }
        k
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
            let (c, lines, more) = s.fetch(&key(), cursor, &Filter::None);
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
        s.set_known(vec![release("default", known_of(&[M1]))]);
        // A release record is published as a seed, in the log, but a client's copy is not new.
        assert_eq!(published(&s), [M1]);
        let o = s.submit(&key(), 1, 1, &recs(&[M1]), 100);
        assert_eq!((o.counted, o.published), (0, 0));
        // Its module satisfies the pipelines.
        let o = s.submit(&key(), 1, 1, &recs(&[M2, P12]), 100);
        assert_eq!((o.counted, o.published), (2, 2));
        assert_eq!(published(&s), [M1, M2, P12]);
    }

    #[test]
    fn known_loaded_after_pending_publishes_them() {
        let mut s = store(2);
        s.submit(&key(), 1, 1, &recs(&[M1, P12]), 100);
        s.submit(&key(), 2, 2, &recs(&[P12]), 100); // ready, waiting for modules
        assert_eq!(s.counts(&key()), (0, 2));
        s.set_known(vec![release("default", known_of(&[M1, M2]))]);
        // The pending M1 became a seed; the release's M2 and the waiting P12 follow.
        assert_eq!(published(&s), [M1, M2, P12]);
        assert_eq!(s.counts(&key()), (3, 0));
    }

    #[test]
    fn releases_are_published_in_order_and_once() {
        let mut s = store(1);
        let m3 = "M 000000000003 ps code=cc";
        let p13 = "P v=000000000001 p=000000000003 plain";
        // v19 ships a pipeline before its module (file order) and a second release ships more.
        s.set_known(vec![
            release("v19", known_of(&[P12, M1, M2, T1])),
            release("v20", known_of(&[M1, m3, p13])),
        ]);
        assert_eq!(published(&s), [M1, M2, T1, P12, m3, p13]);
        // Reloading the same releases publishes nothing again.
        s.set_known(vec![
            release("v19", known_of(&[P12, M1, M2, T1])),
            release("v20", known_of(&[M1, m3, p13])),
        ]);
        assert_eq!(published(&s).len(), 6);
        // A pipeline whose module is in no release never goes out; the rest does.
        let mut s = store(1);
        s.set_known(vec![release("v19", known_of(&[P12, M1]))]);
        assert_eq!(published(&s), [M1]);
        // Uploads of release lines are not new.
        let o = s.submit(&key(), 1, 1, &recs(&[M1, P12]), 100);
        assert_eq!((o.counted, o.published), (0, 0));
    }

    fn two_releases() -> Store {
        let mut s = store(1);
        let m3 = "M 000000000003 ps code=cc";
        s.set_known(vec![release("v19", known_of(&[M1, M2])), release("v20", known_of(&[M1, M2, P12, m3]))]);
        s.submit(&key(), 1, 1, &recs(&[T1]), 100);
        s
    }

    fn fetch_all(s: &Store, filter: &Filter) -> Vec<String> {
        let mut out = Vec::new();
        let mut cursor = 0;
        loop {
            let (c, lines, more) = s.fetch(&key(), cursor, filter);
            out.extend(lines.iter().map(|l| l.to_string()));
            cursor = c;
            if !more {
                return out;
            }
        }
    }

    #[test]
    fn default_release_filter() {
        let mut s = two_releases();
        s.settings.default_release = "v19".into();
        // Nothing of the default release; what only v20 ships and what the players sent, yes.
        assert_eq!(fetch_all(&s, &Filter::DefaultRelease), [P12, "M 000000000003 ps code=cc", T1]);
        assert_eq!(fetch_all(&s, &Filter::None).len(), 5);
        // A default release that does not exist (or has no file for the partition) filters nothing.
        s.settings.default_release = "v21".into();
        assert_eq!(fetch_all(&s, &Filter::DefaultRelease).len(), 5);
        s.settings.default_release = "v19".into();
        let other = PartKey::new(FMT, "PPSA21564", "01.008.000");
        s.submit(&other, 1, 1, &recs(&[M1]), 100);
        assert_eq!(s.fetch(&other, 0, &Filter::DefaultRelease).1.len(), 1);
    }

    #[test]
    fn have_filter() {
        let s = two_releases();
        let have: HashSet<u64> = [M1, P12, T1].iter().map(|l| record_hash(l)).collect();
        assert_eq!(fetch_all(&s, &Filter::Have(have)), [M2, "M 000000000003 ps code=cc"]);
        // An empty `have` gets everything.
        assert_eq!(fetch_all(&s, &Filter::Have(HashSet::new())).len(), 5);
        assert_eq!(parse_hash("00000000000000ff"), Some(255));
        assert_eq!(parse_hash("ff"), None);
        assert_eq!(parse_hash("zz00000000000000"), None);
    }

    #[test]
    fn filtered_pages_review_page_size_lines() {
        // page_size is 3: nine published, the client has the first eight.
        let mut s = store(1);
        let lines: Vec<String> = (1..=9).map(|i| format!("T {i:016x} x")).collect();
        s.submit(&key(), 1, 1, &strings(&lines), 100);
        let have: HashSet<u64> = lines[..8].iter().map(|l| record_hash(l)).collect();
        let filter = Filter::Have(have);
        // The first two pages are entirely left out: empty, cursor advances, more is set.
        let (c, got, more) = s.fetch(&key(), 0, &filter);
        assert_eq!((c, got.len(), more), (3, 0, true));
        let (c, got, more) = s.fetch(&key(), c, &filter);
        assert_eq!((c, got.len(), more), (6, 0, true));
        let (c, got, more) = s.fetch(&key(), c, &filter);
        assert_eq!((c, got.len(), more), (9, 1, false));
        assert_eq!(&*got[0], lines[8]);
    }

    #[test]
    fn cursor_and_more() {
        let mut s = store(1);
        let lines: Vec<String> = (1..=7).map(|i| format!("T {i:016x} x")).collect();
        s.submit(&key(), 1, 1, &strings(&lines), 100);
        let (c, got, more) = s.fetch(&key(), 0, &Filter::None);
        assert_eq!((c, got.len(), more), (3, 3, true));
        let (c, got, more) = s.fetch(&key(), c, &Filter::None);
        assert_eq!((c, got.len(), more), (6, 3, true));
        assert_eq!(&*got[0], lines[3]);
        let (c, got, more) = s.fetch(&key(), c, &Filter::None);
        assert_eq!((c, got.len(), more), (7, 1, false));
        // Nothing new: the cursor stays and the list is empty.
        let (c, got, more) = s.fetch(&key(), c, &Filter::None);
        assert_eq!((c, got.len(), more), (7, 0, false));
        // A cursor beyond the end (server restarted empty) starts over.
        let (c, got, _) = s.fetch(&key(), 99, &Filter::None);
        assert_eq!((c, got.len()), (3, 3));
        assert_eq!(s.fetch(&PartKey::new(FMT, "PPSA00000", "1"), 5, &Filter::None), (0, vec![], false));
    }

    #[test]
    fn partitions_are_independent() {
        let mut s = store(1);
        let other = PartKey::new(FMT, "PPSA21564", "01.008.000");
        s.submit(&key(), 1, 1, &recs(&[M1]), 100);
        s.submit(&other, 1, 1, &recs(&[M2]), 100);
        assert_eq!(published(&s), [M1]);
        assert_eq!(s.fetch(&other, 0, &Filter::None).1.len(), 1);
    }

    #[test]
    fn limits_of_partition_budget_and_count() {
        let mut s = Store::with_salt(
            Settings { quorum: 1, partition_cap: 3, max_partitions: 1, page_size: 10, ..Default::default() },
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
        s.set_known(vec![release("default", known)]);
        // Published: M1 (two clients). Pending: M2 (one client), P12 ready but waiting for M2.
        s.submit(&key(), 1, 1, &recs(&[M1, M2, P12, T1]), 100);
        s.submit(&key(), 2, 2, &recs(&[M1, P12]), 100);
        assert_eq!(s.counts(&key()), (2, 2));
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
        assert_eq!(published(&r), [T1, M1]);
        assert_eq!(r.counts(&key()), (2, 2));
        // The release is loaded again at every start: nothing is published twice.
        r.set_known(vec![release("default", known_of(&[T1]))]);
        assert_eq!(published(&r), [T1, M1]);
        // The pending state survived: a second client completes M2 and releases P12.
        r.submit(&key(), 1, 1, &recs(&[M2]), 100);
        let o = r.submit(&key(), 3, 3, &recs(&[M2]), 100);
        assert_eq!(o.published, 2);
        assert_eq!(published(&r), [T1, M1, M2, P12]);
        r.save(&dir).unwrap();
        let text = fs::read_to_string(export.join("PPSA21564_01.007.000.kpm")).unwrap();
        assert_eq!(text, format!("{FMT}\n{M1}\n{M2}\n{P12}\n"));
        assert!(!export.join("PPSA21564_01.007.000.pending.kpm").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_known_reads_releases_in_folders() {
        let dir = std::env::temp_dir().join(format!("pss-releases-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        for r in ["v10", "v9", "v19", ".hidden"] {
            fs::create_dir_all(dir.join(r)).unwrap();
        }
        fs::write(dir.join("v9").join("PPSA21564_01.007.000.kpm"), format!("{FMT}\n{M1}\n")).unwrap();
        fs::write(dir.join("v10").join("PPSA21564_01.007.000.kpm"), format!("{FMT}\n{M1}\n{M2}\n")).unwrap();
        fs::write(dir.join("v19").join("PPSA21564_01.008.000.kpm"), format!("{FMT}\n{T1}\n")).unwrap();
        fs::write(dir.join(".hidden").join("PPSA21564_01.007.000.kpm"), format!("{FMT}\n{M1}\n")).unwrap();
        let releases = load_known(&dir).unwrap();
        let names: Vec<&str> = releases.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, ["v9", "v10", "v19"], "natural order, no loose files, no hidden folders");
        // The loose files of the older layout are the release "default", first.
        fs::write(dir.join("PPSA21564_01.007.000.kpm"), format!("{FMT}\n{M2}\n")).unwrap();
        let releases = load_known(&dir).unwrap();
        assert_eq!(releases[0].name, "default");
        assert_eq!(releases.len(), 4);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn state_of_the_first_server_loads_with_both_layouts() {
        // A state.json as the version that only knew one folder wrote it: M1 and T1 published
        // (T1 also ships in the repository), P12 pending.
        let dir = std::env::temp_dir().join(format!("pss-migrate-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let state = serde_json::json!({
            "version": 1,
            "salt": "07070707070707070707070707070707",
            "partitions": [{
                "format": FMT, "title": "PPSA21564", "version": "01.007.000",
                "published": [M1, T1],
                "pending": [{"line": P12, "pairs": [[1, 2]], "ready": false}],
            }],
        });
        fs::write(dir.join("state.json"), state.to_string()).unwrap();
        for layout in ["loose", "folder"] {
            let known = dir.join(format!("known-{layout}"));
            let files = if layout == "loose" { known.clone() } else { known.join("v19") };
            fs::create_dir_all(&files).unwrap();
            fs::write(files.join("PPSA21564_01.007.000.kpm"), format!("{FMT}\n{T1}\n{M2}\n")).unwrap();
            let mut s = Store::load(Settings { quorum: 2, page_size: 10, ..Default::default() }, &dir).unwrap();
            assert_eq!(published(&s), [M1, T1]);
            s.set_known(load_known(&known).unwrap());
            // Nothing lost; the release's M2 is added once; the pending pipeline is still pending.
            assert_eq!(published(&s), [M1, T1, M2]);
            assert_eq!(s.counts(&key()), (3, 1));
            // A second client confirms the pipeline (its modules are all published now).
            assert_eq!(s.submit(&key(), 2, 3, &recs(&[P12]), 100).published, 1);
            assert_eq!(published(&s), [M1, T1, M2, P12]);
        }
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
        let releases = load_known(&dir).unwrap();
        assert_eq!(releases.len(), 1);
        assert_eq!(releases[0].name, "default");
        let known = &releases[0].parts;
        assert_eq!(known.len(), 1);
        let k = &known[&key()];
        assert_eq!(k.len(), 2);
        assert_eq!(&*k.order[0], M1);
        let _ = fs::remove_dir_all(&dir);
    }

    // A TES module, and a pipeline that names it as its vertex stage: a recording that mixed up
    // its modules wrote such records (Elden Ring Nightreign crashed creating one).
    const MT: &str = "M 000000000004 tes code=dd ticp=3";
    const PBAD: &str = "P v=000000000004 p=000000000002 plain";

    #[test]
    fn unlinkable_pipelines_are_left_out_of_uploads() {
        let mut s = store(1);
        s.submit(&key(), 1, 1, &recs(&[M1, M2, MT]), 100);
        // Its modules are published: refused at once, not counted against the budget.
        let o = s.submit(&key(), 1, 1, &recs(&[PBAD, P12]), 100);
        assert_eq!((o.counted, o.published, o.unlinked.count), (1, 1, 1));
        assert_eq!(o.unlinked.example.as_deref(), Some(&*format!("{PBAD} (stages tes,-,-,ps)")));
        assert_eq!(published(&s), [M1, M2, MT, P12]);
        assert_eq!(s.counts(&key()), (4, 0));
        // Before its modules in one request: it waits, and goes when they arrive.
        let mut s = store(1);
        let o = s.submit(&key(), 1, 1, &recs(&[PBAD, M1, M2, MT]), 100);
        assert_eq!((o.published, o.unlinked.count), (3, 1));
        assert_eq!(s.counts(&key()), (3, 0));
        // Without quorum it is not kept as pending either.
        let mut s = store(2);
        s.set_known(vec![release("default", known_of(&[M1, M2, MT]))]);
        let o = s.submit(&key(), 1, 1, &recs(&[PBAD]), 100);
        assert_eq!((o.counted, o.unlinked.count), (0, 1));
        assert_eq!(s.counts(&key()), (3, 0));
    }

    #[test]
    fn seeds_leave_out_unlinkable_pipelines() {
        let mut s = store(1);
        let rejected = s.set_known(vec![release("default", known_of(&[PBAD, M1, M2, MT, P12]))]);
        assert_eq!(published(&s), [M1, M2, MT, P12]);
        assert_eq!(rejected.len(), 1);
        assert_eq!((&rejected[0].0, rejected[0].1.count), (&key(), 1));
        // Loading the release again leaves it out again, and publishes nothing twice.
        let rejected = s.set_known(vec![release("default", known_of(&[PBAD, M1, M2, MT, P12]))]);
        assert_eq!(rejected[0].1.count, 1);
        assert_eq!(published(&s).len(), 4);
    }

    #[test]
    fn state_with_unlinkable_pipelines_is_cleaned() {
        // A state saved by a server that did not check pipelines: PBAD published (4th), another
        // bad one pending, and a pipeline that waits for a module nobody sent.
        let dir = std::env::temp_dir().join(format!("pss-unlinked-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let pbad2 = "P v=000000000002";
        let pwait = "P v=000000000009 p=000000000002";
        let state = serde_json::json!({
            "version": 1,
            "salt": "07070707070707070707070707070707",
            "partitions": [{
                "format": FMT, "title": "PPSA21564", "version": "01.007.000",
                "published": [M1, M2, MT, PBAD, P12, T1],
                "pending": [{"line": pbad2, "pairs": [[1, 2]], "ready": false},
                            {"line": pwait, "pairs": [[1, 2]], "ready": true}],
            }],
        });
        fs::write(dir.join("state.json"), state.to_string()).unwrap();
        let settings = || Settings { quorum: 2, page_size: 10, ..Default::default() };
        let mut s = Store::load(settings(), &dir).unwrap();
        let removed = s.drop_unlinkable();
        assert_eq!(removed.len(), 1);
        assert_eq!((&removed[0].0, removed[0].1, removed[0].2, removed[0].3.count), (&key(), 1, 1, 2));
        assert!(s.is_dirty());
        // Not served, and the records after it keep their place: a client that had read up to
        // PBAD (cursor 4) gets the rest.
        assert_eq!(published(&s), [M1, M2, MT, P12, T1]);
        assert_eq!(s.fetch(&key(), 0, &Filter::None).0, 6);
        let (c, lines, more) = s.fetch(&key(), 4, &Filter::None);
        assert_eq!((c, lines.iter().map(|l| l.to_string()).collect::<Vec<_>>(), more), (6, vec![P12.to_string(), T1.into()], false));
        assert_eq!(s.counts(&key()), (5, 1));
        // A client that sends it again is refused.
        assert_eq!(s.submit(&key(), 3, 3, &recs(&[PBAD]), 100).unlinked.count, 1);
        // Saved as a tombstone and not exported; a restart finds nothing more to remove.
        s.save(&dir).unwrap();
        let text = fs::read_to_string(dir.join("export").join("v1").join("PPSA21564_01.007.000.kpm")).unwrap();
        assert_eq!(text, format!("{FMT}\n{}\n", [M1, M2, MT, P12, T1].join("\n")));
        let saved: serde_json::Value = serde_json::from_str(&fs::read_to_string(dir.join("state.json")).unwrap()).unwrap();
        assert_eq!(saved["partitions"][0]["published"][3], "");
        let mut r = Store::load(settings(), &dir).unwrap();
        assert!(r.drop_unlinkable().is_empty());
        assert_eq!(published(&r), [M1, M2, MT, P12, T1]);
        assert_eq!(r.fetch(&key(), 0, &Filter::None).0, 6);
        let _ = fs::remove_dir_all(&dir);
    }

    // ---- Copies between the partitions of one game ----

    fn game_store(quorum: usize, min_codes: usize) -> Store {
        let regions = Regions::parse("Game: PPSA00001 PPSA00002\nOther: PPSA00008 PPSA00009").unwrap();
        let merge = Merge { regions, share_percent: 50, min_codes };
        Store::with_salt(Settings { quorum, page_size: 1000, merge: Some(merge), ..Default::default() }, [7; 16])
    }

    fn part(title: &str, version: &str) -> PartKey {
        PartKey::new(FMT, title, version)
    }

    /// A vertex module (`vs(n)`), a pixel module (`ps(n)`) and a pipeline of both.
    fn vs(n: u32) -> String {
        format!("M {:012x} vs code={:016x}", 0x1000 + n, 0x1000 + n)
    }

    fn ps(n: u32) -> String {
        format!("M {:012x} ps code={:016x}", 0x2000 + n, 0x2000 + n)
    }

    fn pipe(v: u32, p: u32) -> String {
        format!("P v={:012x} p={:012x}", 0x1000 + v, 0x2000 + p)
    }

    fn all(s: &Store, key: &PartKey, filter: &Filter) -> Vec<String> {
        let mut out = Vec::new();
        let mut cursor = 0;
        loop {
            let (c, lines, more) = s.fetch(key, cursor, filter);
            out.extend(lines.iter().map(|l| l.to_string()));
            cursor = c;
            if !more {
                return out;
            }
        }
    }

    fn send(s: &mut Store, key: &PartKey, client: u64, lines: &[String]) -> SubmitOutcome {
        s.submit(key, client, client, &strings(lines), 1000)
    }

    #[test]
    fn regions_of_one_version_share_their_records() {
        let mut s = game_store(1, 4);
        let (a, b, c) = (part("PPSA00001", "01.000"), part("PPSA00002", "01.000"), part("PPSA00008", "01.000"));
        send(&mut s, &a, 1, &[vs(0), ps(0), pipe(0, 0), T1.to_string()]);
        send(&mut s, &c, 1, &[vs(9)]);
        // The first record of the other region links the two: each gets what the other has, the
        // modules before the pipelines, and no tessellation records.
        let o = send(&mut s, &b, 2, &[vs(1)]);
        assert_eq!((o.published, o.copied), (1, 4));
        assert_eq!(all(&s, &b, &Filter::None), [vs(1), vs(0), ps(0), pipe(0, 0)]);
        assert_eq!(all(&s, &a, &Filter::None), [vs(0), ps(0), pipe(0, 0), T1.to_string(), vs(1)]);
        assert_eq!((s.copies(&a), s.copies(&b)), (1, 3));
        assert_eq!(all(&s, &c, &Filter::None), [vs(9)], "another game");
        // What is published later goes over at once, in order.
        let o = send(&mut s, &a, 1, &[pipe(2, 2), vs(2), ps(2)]);
        assert_eq!((o.published, o.copied), (3, 3));
        assert_eq!(all(&s, &b, &Filter::None)[4..], [vs(2), ps(2), pipe(2, 2)]);
        // A copy is no client of the partition: its own clients make it its own, with quorum. b's
        // first record waits for its quorum, but b gets a's at once.
        let mut s2 = game_store(2, 4);
        send(&mut s2, &a, 1, &[vs(0)]);
        send(&mut s2, &a, 2, &[vs(0)]);
        send(&mut s2, &b, 3, &[vs(1)]);
        send(&mut s2, &b, 4, &[vs(1)]);
        assert_eq!((s2.copies(&b), s2.counts(&b)), (1, (2, 0)));
        let o = send(&mut s2, &b, 3, &[vs(0)]);
        assert_eq!((o.counted, o.published, s2.copies(&b)), (1, 0, 1));
        let o = send(&mut s2, &b, 4, &[vs(0)]);
        assert_eq!((o.counted, o.published, s2.copies(&b)), (1, 0, 0));
        assert_eq!(all(&s2, &b, &Filter::None), [vs(0), vs(1)], "it keeps its place");
        // A record pending in the partition is published as a copy and keeps its client.
        send(&mut s2, &b, 3, &[ps(5)]);
        assert_eq!(s2.counts(&b), (2, 1));
        send(&mut s2, &a, 1, &[ps(5)]);
        send(&mut s2, &a, 2, &[ps(5)]);
        assert_eq!((s2.counts(&b), s2.copies(&b)), ((3, 0), 1));
        send(&mut s2, &b, 4, &[ps(5)]);
        assert_eq!(s2.copies(&b), 0, "the pending client counts for the quorum");
    }

    #[test]
    fn a_client_that_sends_nothing_gets_the_other_region() {
        let mut s = game_store(1, 4);
        let (a, b) = (part("PPSA00001", "01.000"), part("PPSA00002", "01.000"));
        send(&mut s, &a, 1, &[vs(0)]);
        let o = s.submit(&b, 2, 2, &[], 1000);
        assert_eq!((o.copied, s.partitions()), (1, 2));
        assert_eq!(all(&s, &b, &Filter::None), [vs(0)]);
        // Not for a version no other region has, nor for another game.
        s.submit(&part("PPSA00002", "01.001"), 2, 2, &[], 1000);
        s.submit(&part("PPSA00009", "01.000"), 2, 2, &[], 1000);
        assert_eq!(s.partitions(), 2);
    }

    #[test]
    fn versions_share_their_records_only_with_enough_shaders_in_common() {
        let mut s = game_store(1, 4);
        let old = part("PPSA00001", "01.000");
        let new = part("PPSA00001", "01.001");
        let far = part("PPSA00002", "02.000");
        send(&mut s, &old, 1, &(0..10).map(vs).collect::<Vec<_>>());
        // Three shaders, all in the old version: too few to tell.
        send(&mut s, &new, 2, &[vs(0), vs(1), vs(2)]);
        assert_eq!(s.copies(&new), 0);
        // The fourth links them: the new version gets the rest of the old one.
        let o = send(&mut s, &new, 2, &[vs(3)]);
        assert_eq!((o.copied, s.copies(&new), s.copies(&old)), (6, 6, 0));
        // Another region and version with 3 of its 8 shaders in common (37.5 %): apart.
        let lines: Vec<String> = (7..15).map(vs).collect();
        let o = send(&mut s, &far, 3, &lines);
        assert_eq!((o.copied, s.copies(&far)), (0, 0));
        // Once linked, a pair stays so even if the share falls under the threshold.
        send(&mut s, &new, 2, &(100..140).map(ps).collect::<Vec<_>>());
        assert_eq!(s.share(&old, &new), (4, 10));
        let o = send(&mut s, &old, 1, &[ps(1)]);
        assert_eq!(o.copied, 1);
        let reports = s.merge();
        assert_eq!(reports.len(), 1);
        let r = &reports[0];
        assert_eq!((r.game.as_str(), r.copies, r.added), ("Game", 7 + 40, 0));
        assert_eq!(
            r.to_string(),
            "merge Game (v1): together PPSA00001_01.000 + PPSA00001_01.001 36.4% (4 of 11); \
             apart PPSA00001_01.000 + PPSA00002_02.000 37.5% (3 of 8), PPSA00001_01.001 + PPSA00002_02.000 0.0% (0 of 8); \
             copies 47 (0 new)"
        );
    }

    #[test]
    fn copies_bring_modules_before_the_pipelines_that_name_them() {
        let mut s = game_store(1, 4);
        let a = part("PPSA00001", "1");
        let b = part("PPSA00002", "1");
        let c = part("PPSA00001", "2");
        send(&mut s, &a, 1, &(0..5).map(vs).collect::<Vec<_>>());
        send(&mut s, &b, 2, &(20..25).map(vs).collect::<Vec<_>>());
        let mut lines: Vec<String> = (0..5).map(vs).collect();
        lines.push(ps(9));
        send(&mut s, &c, 3, &lines);
        // a is linked with b (one version) and c (all of its shaders), b and c are apart: the copy
        // of c's module in a does not go on to b.
        assert_eq!(s.linked(&a), [c.clone(), b.clone()]);
        assert!(s.linked(&b) == [a.clone()] && !all(&s, &b, &Filter::None).contains(&ps(9)));
        // A player of a records a pipeline with that module: the module is its own now (its
        // export needs it) and b gets it before the pipeline (c gets the pipeline).
        let o = send(&mut s, &a, 1, &[pipe(0, 9)]);
        assert_eq!((o.published, o.copied), (1, 3));
        let got = all(&s, &b, &Filter::None);
        assert_eq!(got[got.len() - 2..], [ps(9), pipe(0, 9)]);
        // A pipeline that the destination cannot link is not copied: module 0x1000 is a pixel
        // stage there (a different record with that id).
        let mut s = game_store(1, 4);
        let odd = format!("M {:012x} ps code=0000000000000abc", 0x1000);
        send(&mut s, &b, 2, &[odd.clone(), ps(0)]);
        send(&mut s, &a, 1, &[vs(0), ps(0), pipe(0, 0)]);
        assert_eq!(all(&s, &b, &Filter::None), [odd, ps(0)]);
        assert_eq!(s.copies(&b), 0);
    }

    #[test]
    fn copies_respect_tombstones_and_the_partition_cap() {
        // b removed an unlinkable pipeline (a tombstone at 2); a has the same line, linkable there.
        let dir = std::env::temp_dir().join(format!("pss-copy-tomb-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let state = serde_json::json!({
            "version": 1,
            "salt": "07070707070707070707070707070707",
            "partitions": [
                {"format": FMT, "title": "PPSA00001", "version": "1", "published": ["M 000000000004 vs code=dd", M2, PBAD], "pending": []},
                {"format": FMT, "title": "PPSA00002", "version": "1", "published": [MT, M2, PBAD], "pending": []},
            ],
        });
        fs::write(dir.join("state.json"), state.to_string()).unwrap();
        let mut s = Store::load(game_store(1, 4).settings.clone(), &dir).unwrap();
        s.drop_unlinkable();
        s.merge();
        let b = part("PPSA00002", "1");
        // No copy in the tombstone's place and the pipeline does not come back.
        assert_eq!(s.parts[&b].published.iter().map(|l| l.to_string()).collect::<Vec<_>>(), [MT, M2, ""]);
        assert_eq!(s.copies(&b), 0);
        let _ = fs::remove_dir_all(&dir);

        // Copies have a limit of their own: they do not take the room of the own records.
        let mut s = game_store(1, 4);
        s.settings.partition_cap = 3;
        let (a, b) = (part("PPSA00001", "1"), part("PPSA00002", "1"));
        send(&mut s, &a, 1, &(0..3).map(vs).collect::<Vec<_>>());
        send(&mut s, &b, 2, &[ps(0)]);
        assert_eq!(s.copies(&b), 3);
        let o = send(&mut s, &b, 2, &[ps(1), ps(2), ps(3)]);
        assert_eq!((o.published, o.dropped), (2, 1));
        assert_eq!(s.copies(&a), 3, "a holds three copies of b's");
    }

    #[test]
    fn export_leaves_out_copies_and_the_state_reloads() {
        let dir = std::env::temp_dir().join(format!("pss-copy-state-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut s = game_store(1, 4);
        let (a, b) = (part("PPSA00001", "1"), part("PPSA00002", "1"));
        let v19 = || vec![Release { name: "v19".into(), parts: HashMap::from([(part("PPSA00001", "1"), known_of(&[&vs(0)]))]) }];
        s.set_known(v19());
        send(&mut s, &a, 1, &[ps(0), pipe(0, 0)]);
        // b's pipeline names a's module ps(0): it waits for it, and the copy makes it its own.
        send(&mut s, &b, 2, &[vs(1), pipe(1, 0)]);
        let b_all = all(&s, &b, &Filter::None);
        assert_eq!(b_all, [vs(1), vs(0), ps(0), pipe(0, 0), pipe(1, 0)]);
        assert_eq!(all(&s, &a, &Filter::None), [vs(0), ps(0), pipe(0, 0), vs(1), pipe(1, 0)]);
        assert_eq!((s.copies(&a), s.copies(&b)), (2, 2));
        // The first clients (v19) get the copies too: their release does not ship them.
        s.settings.default_release = "v19".into();
        assert_eq!(all(&s, &b, &Filter::DefaultRelease), b_all);
        s.save(&dir).unwrap();
        let export = dir.join("export").join("v1");
        let read = |name: &str| fs::read_to_string(export.join(name)).unwrap();
        // Own records only, without what the release ships; ps(0) is b's own now.
        assert_eq!(read("PPSA00001_1.kpm"), format!("{FMT}\n{}\n{}\n", ps(0), pipe(0, 0)));
        assert_eq!(read("PPSA00002_1.kpm"), format!("{FMT}\n{}\n{}\n{}\n", vs(1), ps(0), pipe(1, 0)));
        // In state.json a copy is a tombstone in its place plus an entry apart.
        let saved: serde_json::Value = serde_json::from_str(&fs::read_to_string(dir.join("state.json")).unwrap()).unwrap();
        let pb = &saved["partitions"][1];
        assert_eq!(pb["published"], serde_json::json!([vs(1), "", ps(0), "", pipe(1, 0)]));
        assert_eq!(pb["copies"], serde_json::json!([{"at": 1, "line": vs(0)}, {"at": 3, "line": pipe(0, 0)}]));
        assert_eq!(saved["links"], serde_json::json!([[FMT, "PPSA00001", "1", "PPSA00002", "1"]]));

        // A server with copies: the same order, the same copies, nothing new.
        let mut r = Store::load(s.settings.clone(), &dir).unwrap();
        r.set_known(v19());
        assert_eq!(r.merge()[0].added, 0);
        assert_eq!(all(&r, &b, &Filter::None), b_all);
        assert_eq!((r.copies(&a), r.copies(&b), r.links.len()), (2, 2, 1));
        r.save(&dir).unwrap();
        assert_eq!(read("PPSA00002_1.kpm"), format!("{FMT}\n{}\n{}\n{}\n", vs(1), ps(0), pipe(1, 0)));

        // A server without copies reads tombstones there (it serves and exports none of them);
        // when the one with copies comes back without them, it copies them again after the end
        // and the cursors keep their records.
        let mut old = saved.clone();
        for p in old["partitions"].as_array_mut().unwrap() {
            p.as_object_mut().unwrap().remove("copies");
        }
        old.as_object_mut().unwrap().remove("links");
        fs::write(dir.join("state.json"), old.to_string()).unwrap();
        let mut r = Store::load(s.settings.clone(), &dir).unwrap();
        assert_eq!(all(&r, &b, &Filter::None), [vs(1), ps(0), pipe(1, 0)]);
        assert_eq!(r.merge()[0].added, 4);
        assert_eq!(all(&r, &b, &Filter::None), [vs(1), ps(0), pipe(1, 0), vs(0), pipe(0, 0)]);
        let rest: Vec<String> = r.fetch(&b, 5, &Filter::None).1.iter().map(|l| l.to_string()).collect();
        assert_eq!(rest, [vs(0), pipe(0, 0)]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn without_merge_nothing_is_copied() {
        let mut s = game_store(1, 4);
        s.settings.merge = None;
        let (a, b) = (part("PPSA00001", "1"), part("PPSA00002", "1"));
        send(&mut s, &a, 1, &[vs(0)]);
        send(&mut s, &b, 2, &[vs(1)]);
        assert!(s.merge().is_empty());
        assert_eq!((s.copies(&b), all(&s, &b, &Filter::None)), (0, vec![vs(1)]));
        assert!(s.parts.values().all(|p| p.fresh.is_empty()));
    }

    #[test]
    fn stats_count_own_records_the_release_and_copies() {
        let mut s = game_store(1, 4);
        let (a, b) = (part("PPSA00001", "1"), part("PPSA00002", "1"));
        // Two modules of one shader count as one shader.
        let variant = format!("M {:012x} vs code={:016x}", 0x3000, 0x1000);
        send(&mut s, &a, 1, &[vs(0), variant.clone(), ps(0), pipe(0, 0), T1.to_string()]);
        send(&mut s, &b, 2, &[vs(1)]);
        // The latest release ships vs(0) and the pipeline of a; it is compared with, never published.
        let v20 = Release { name: "v20".into(), parts: HashMap::from([(a.clone(), known_of(&[&vs(0), &pipe(0, 0), &vs(7)]))]) };
        s.set_reference(Some(v20));
        assert_eq!(s.reference_name(), Some("v20"));
        assert!(!all(&s, &a, &Filter::None).contains(&vs(7)));
        let stats = s.stats();
        let got: Vec<(String, [usize; 6])> = stats
            .iter()
            .map(|p| {
                let n = [p.shaders, p.shaders_in_release, p.pipelines, p.pipelines_in_release, p.copied_shaders, p.copied_pipelines];
                (p.key.to_string(), n)
            })
            .collect();
        // a: 2 shaders (1 in the release), 1 pipeline (in it), b's shader as a copy. b: its shader,
        // and a's two shaders (three modules) and pipeline as copies.
        assert_eq!(got, [("PPSA00001_1".to_string(), [2, 1, 1, 1, 1, 0]), ("PPSA00002_1".to_string(), [1, 0, 0, 0, 2, 1])]);
        // A partition with nothing published is not listed.
        let mut s = store(2);
        s.submit(&key(), 1, 1, &recs(&[M1]), 100);
        assert!(s.stats().is_empty());
    }
}
