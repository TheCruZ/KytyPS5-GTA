//! Seeds: the `.kpm` manifests of the repository, downloaded from GitHub at every start and kept
//! on disk. The server never needs a copy by hand; when GitHub cannot be reached it uses the
//! copy of the last start.

use sha1::{Digest as _, Sha1};
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Marks a folder of a tag (immutable) that was downloaded completely: it is never fetched again.
const COMPLETE: &str = ".complete";
const MAX_FILE_BYTES: u64 = 256 * 1024 * 1024;

/// Blocking HTTP GET, behind a trait so that the tests do not need a network.
pub trait Fetcher {
    fn get(&self, url: &str, timeout: Duration) -> Result<Vec<u8>, String>;
}

/// The real thing: HTTPS with rustls and the root certificates embedded in the binary.
pub struct HttpFetcher;

impl Fetcher for HttpFetcher {
    fn get(&self, url: &str, timeout: Duration) -> Result<Vec<u8>, String> {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            .user_agent(concat!("pipeline-share-server/", env!("CARGO_PKG_VERSION")))
            .build()
            .into();
        let mut response = agent
            .get(url)
            .header("Accept", "application/vnd.github+json")
            .call()
            .map_err(|e| e.to_string())?;
        response
            .body_mut()
            .with_config()
            .limit(MAX_FILE_BYTES)
            .read_to_vec()
            .map_err(|e| e.to_string())
    }
}

#[derive(Clone, Debug)]
pub struct Config {
    /// `https://api.github.com`
    pub api: String,
    /// `owner/name`
    pub repo: String,
    /// Branch or tag whose manifests are the seeds, e.g. `main`.
    pub git_ref: String,
    /// Folder of the repository with the `.kpm` files.
    pub path: String,
    /// A tag whose manifests are fetched once (the default release), if it has no folder yet.
    pub tag: Option<String>,
    /// Where the copies are kept: one folder per ref.
    pub dir: PathBuf,
    /// Total time for all the downloads of a start.
    pub timeout: Duration,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub downloaded: usize,
    pub cached: usize,
    pub failed: usize,
}

#[derive(serde::Deserialize)]
struct Entry {
    name: String,
    sha: String,
    #[serde(rename = "type")]
    kind: String,
    download_url: Option<String>,
}

/// Names the folder of a ref: `feature/x` is kept as `feature_x`.
pub fn folder_name(git_ref: &str) -> String {
    git_ref.replace('/', "_")
}

/// `owner/name`, a ref or a path: letters, digits and `._-/`, no `..`.
pub fn valid_component(s: &str) -> bool {
    !s.is_empty()
        && !s.contains("..")
        && s.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b'/'))
}

fn file_name_ok(name: &str) -> bool {
    name.ends_with(".kpm")
        && name.len() <= 128
        && !name.starts_with('.')
        && name.bytes().all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}

/// The id Git gives a file: SHA-1 of `blob <size>\0<content>`.
pub fn git_blob_sha(data: &[u8]) -> String {
    let mut h = Sha1::new();
    h.update(format!("blob {}\0", data.len()).as_bytes());
    h.update(data);
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

fn write_atomic(path: &Path, data: &[u8]) -> io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let mut f = fs::File::create(&tmp)?;
    f.write_all(data)?;
    f.sync_all()?;
    drop(f);
    fs::rename(&tmp, path)
}

fn remaining(deadline: Instant) -> Option<Duration> {
    let left = deadline.saturating_duration_since(Instant::now());
    (left >= Duration::from_millis(200)).then_some(left)
}

/// Lists the `.kpm` files of `path` at `git_ref` and copies to `dest` the ones that changed (by
/// their Git blob id; the downloaded content is checked against it). Files that are gone from
/// the repository stay in `dest`. Fails only when the listing fails; files that cannot be
/// downloaded keep the copy they had and are counted in `failed`.
pub fn sync_dir(f: &dyn Fetcher, cfg: &Config, git_ref: &str, dest: &Path, deadline: Instant) -> Result<Report, String> {
    let url = format!("{}/repos/{}/contents/{}?ref={}", cfg.api.trim_end_matches('/'), cfg.repo, cfg.path, git_ref);
    let timeout = remaining(deadline).ok_or("out of time")?;
    let listing = f.get(&url, timeout).map_err(|e| format!("listing {url}: {e}"))?;
    let entries: Vec<Entry> = serde_json::from_slice(&listing).map_err(|e| format!("listing {url}: {e}"))?;
    fs::create_dir_all(dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    let mut report = Report::default();
    for entry in entries.iter().filter(|e| e.kind == "file" && file_name_ok(&e.name)) {
        let path = dest.join(&entry.name);
        if fs::read(&path).is_ok_and(|d| git_blob_sha(&d) == entry.sha) {
            report.cached += 1;
            continue;
        }
        let result = (|| {
            let url = entry.download_url.as_deref().filter(|u| u.starts_with("https://") || u.starts_with("http://")).ok_or("no download url")?;
            let timeout = remaining(deadline).ok_or("out of time")?;
            let data = f.get(url, timeout)?;
            if git_blob_sha(&data) != entry.sha {
                return Err("content does not match its blob id".to_string());
            }
            write_atomic(&path, &data).map_err(|e| e.to_string())
        })();
        match result {
            Ok(()) => report.downloaded += 1,
            Err(e) => {
                tracing::warn!("seed {}@{git_ref}: {e}", entry.name);
                report.failed += 1;
            }
        }
    }
    Ok(report)
}

fn has_kpm(dir: &Path) -> bool {
    fs::read_dir(dir).is_ok_and(|rd| rd.flatten().any(|e| e.file_name().to_string_lossy().ends_with(".kpm")))
}

/// Refreshes the copies on disk: the ref (every start) and the tag (only while its folder is
/// not complete). Returns the folders to load, as (release name, folder): the tag first, then
/// the ref. Whatever goes wrong is logged and the copy on disk is used.
pub fn refresh(f: &dyn Fetcher, cfg: &Config) -> Vec<(String, PathBuf)> {
    let deadline = Instant::now() + cfg.timeout;
    let mut out = Vec::new();
    if let Some(tag) = cfg.tag.as_deref().filter(|t| *t != cfg.git_ref) {
        let dest = cfg.dir.join(folder_name(tag));
        if dest.join(COMPLETE).exists() {
            tracing::info!("seeds {tag}: complete copy on disk");
        } else {
            match sync_dir(f, cfg, tag, &dest, deadline) {
                Ok(r) => {
                    tracing::info!("seeds {tag}: {} downloaded, {} cached, {} failed", r.downloaded, r.cached, r.failed);
                    if r.failed == 0 && r.downloaded + r.cached > 0 {
                        let _ = write_atomic(&dest.join(COMPLETE), b"");
                    }
                }
                Err(e) => tracing::warn!("seeds {tag}: {e}; using the copy on disk"),
            }
        }
        if has_kpm(&dest) {
            out.push((folder_name(tag), dest));
        }
    }
    let dest = cfg.dir.join(folder_name(&cfg.git_ref));
    match sync_dir(f, cfg, &cfg.git_ref, &dest, deadline) {
        Ok(r) => tracing::info!(
            "seeds {}: {} downloaded, {} cached, {} failed",
            cfg.git_ref,
            r.downloaded,
            r.cached,
            r.failed
        ),
        Err(e) => tracing::warn!("seeds {}: {e}; using the copy on disk", cfg.git_ref),
    }
    if has_kpm(&dest) {
        out.push((folder_name(&cfg.git_ref), dest));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;

    const API: &str = "https://api.test";

    struct Fake {
        pages: RefCell<HashMap<String, Result<Vec<u8>, String>>>,
        calls: RefCell<Vec<String>>,
    }

    impl Fake {
        fn new() -> Self {
            Self { pages: RefCell::new(HashMap::new()), calls: RefCell::new(Vec::new()) }
        }

        fn put(&self, url: &str, body: Result<Vec<u8>, String>) {
            self.pages.borrow_mut().insert(url.to_string(), body);
        }

        /// A folder of the fake repository at `git_ref` with these (name, content) files.
        fn repo(&self, git_ref: &str, files: &[(&str, &str)]) {
            let list: Vec<serde_json::Value> = files
                .iter()
                .map(|(n, c)| {
                    serde_json::json!({
                        "name": n, "type": "file", "sha": git_blob_sha(c.as_bytes()),
                        "download_url": format!("https://raw.test/{git_ref}/{n}"),
                    })
                })
                .collect();
            self.put(
                &format!("{API}/repos/o/r/contents/assets/pipelines?ref={git_ref}"),
                Ok(serde_json::to_vec(&list).unwrap()),
            );
            for (n, c) in files {
                self.put(&format!("https://raw.test/{git_ref}/{n}"), Ok(c.as_bytes().to_vec()));
            }
        }
    }

    // The trait wants `&self` and a thread-safe fake is not needed: the calls are sequential.
    impl Fetcher for Fake {
        fn get(&self, url: &str, _timeout: Duration) -> Result<Vec<u8>, String> {
            self.calls.borrow_mut().push(url.to_string());
            self.pages.borrow().get(url).cloned().unwrap_or_else(|| Err(format!("404 {url}")))
        }
    }

    fn cfg(dir: &Path, tag: Option<&str>) -> Config {
        Config {
            api: API.into(),
            repo: "o/r".into(),
            git_ref: "main".into(),
            path: "assets/pipelines".into(),
            tag: tag.map(str::to_string),
            dir: dir.to_path_buf(),
            timeout: Duration::from_secs(30),
        }
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("pss-seed-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn git_blob_ids() {
        // `git hash-object` of "hello\n" and of an empty file.
        assert_eq!(git_blob_sha(b"hello\n"), "ce013625030ba8dba906f756967f9e9ca394464a");
        assert_eq!(git_blob_sha(b""), "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391");
    }

    #[test]
    fn lists_downloads_and_caches() {
        let dir = tmp("list");
        let f = Fake::new();
        f.repo("main", &[("PPSA00001_1.kpm", "a\n"), ("PPSA00002_1.kpm", "b\n"), ("README.md", "x"), ("../evil.kpm", "x")]);
        let c = cfg(&dir, None);
        let r = sync_dir(&f, &c, "main", &dir.join("main"), Instant::now() + Duration::from_secs(5)).unwrap();
        assert_eq!(r, Report { downloaded: 2, cached: 0, failed: 0 });
        assert_eq!(fs::read_to_string(dir.join("main/PPSA00001_1.kpm")).unwrap(), "a\n");
        assert!(!dir.join("main/README.md").exists());
        // A second start with nothing changed downloads nothing.
        f.calls.borrow_mut().clear();
        let r = sync_dir(&f, &c, "main", &dir.join("main"), Instant::now() + Duration::from_secs(5)).unwrap();
        assert_eq!(r, Report { downloaded: 0, cached: 2, failed: 0 });
        assert_eq!(f.calls.borrow().len(), 1, "only the listing");
        // One file changed in the repository: only that one is downloaded; a removed one stays.
        f.repo("main", &[("PPSA00001_1.kpm", "a\na2\n")]);
        let r = sync_dir(&f, &c, "main", &dir.join("main"), Instant::now() + Duration::from_secs(5)).unwrap();
        assert_eq!(r, Report { downloaded: 1, cached: 0, failed: 0 });
        assert_eq!(fs::read_to_string(dir.join("main/PPSA00001_1.kpm")).unwrap(), "a\na2\n");
        assert!(dir.join("main/PPSA00002_1.kpm").exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn failures_keep_the_copy_on_disk() {
        let dir = tmp("fail");
        let f = Fake::new();
        f.repo("main", &[("PPSA00001_1.kpm", "a\n"), ("PPSA00002_1.kpm", "b\n")]);
        let c = cfg(&dir, None);
        let seeds = refresh(&f, &c);
        assert_eq!(seeds, [("main".to_string(), dir.join("main"))]);
        // GitHub fails (rate limit): the copy is used and nothing is lost.
        let down = Fake::new();
        down.put(&format!("{API}/repos/o/r/contents/assets/pipelines?ref=main"), Err("403 rate limit".into()));
        let seeds = refresh(&down, &c);
        assert_eq!(seeds, [("main".to_string(), dir.join("main"))]);
        assert_eq!(fs::read_to_string(dir.join("main/PPSA00002_1.kpm")).unwrap(), "b\n");
        // One file fails and another is corrupt: the old content stays.
        f.repo("main", &[("PPSA00001_1.kpm", "a2\n"), ("PPSA00002_1.kpm", "b2\n")]);
        f.put("https://raw.test/main/PPSA00001_1.kpm", Err("timeout".into()));
        f.put("https://raw.test/main/PPSA00002_1.kpm", Ok(b"corrupt\n".to_vec()));
        let r = sync_dir(&f, &c, "main", &dir.join("main"), Instant::now() + Duration::from_secs(5)).unwrap();
        assert_eq!(r, Report { downloaded: 0, cached: 0, failed: 2 });
        assert_eq!(fs::read_to_string(dir.join("main/PPSA00001_1.kpm")).unwrap(), "a\n");
        assert_eq!(fs::read_to_string(dir.join("main/PPSA00002_1.kpm")).unwrap(), "b\n");
        // No copy and no network: nothing to load, no panic.
        let empty = tmp("empty");
        assert!(refresh(&down, &cfg(&empty, Some("v19"))).is_empty());
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&empty);
    }

    #[test]
    fn tag_is_fetched_once() {
        let dir = tmp("tag");
        let f = Fake::new();
        f.repo("v19", &[("PPSA00001_1.kpm", "a\n")]);
        f.repo("main", &[("PPSA00001_1.kpm", "a\nb\n")]);
        let c = cfg(&dir, Some("v19"));
        let seeds = refresh(&f, &c);
        let names: Vec<&str> = seeds.iter().map(|s| s.0.as_str()).collect();
        assert_eq!(names, ["v19", "main"]);
        assert!(dir.join("v19").join(COMPLETE).exists());
        f.calls.borrow_mut().clear();
        refresh(&f, &c);
        assert!(f.calls.borrow().iter().all(|u| !u.contains("ref=v19") && !u.contains("/v19/")), "the tag is not asked again");
        // A tag that failed halfway is completed at the next start.
        let dir2 = tmp("tag2");
        let g = Fake::new();
        g.repo("v19", &[("PPSA00001_1.kpm", "a\n"), ("PPSA00002_1.kpm", "b\n")]);
        g.put("https://raw.test/v19/PPSA00002_1.kpm", Err("reset".into()));
        refresh(&g, &cfg(&dir2, Some("v19")));
        assert!(!dir2.join("v19").join(COMPLETE).exists());
        g.put("https://raw.test/v19/PPSA00002_1.kpm", Ok(b"b\n".to_vec()));
        refresh(&g, &cfg(&dir2, Some("v19")));
        assert!(dir2.join("v19").join(COMPLETE).exists());
        assert!(dir2.join("v19/PPSA00002_1.kpm").exists());
        let _ = fs::remove_dir_all(&dir);
        let _ = fs::remove_dir_all(&dir2);
    }

    #[test]
    fn component_checks() {
        assert!(valid_component("TheCruZ/KytyPS5-GTA"));
        assert!(valid_component("assets/pipelines"));
        assert!(!valid_component("a/../b"));
        assert!(!valid_component("a b"));
        assert!(!valid_component(""));
        assert_eq!(folder_name("feature/x"), "feature_x");
    }

    #[test]
    fn http_fetcher_talks_to_a_local_server() {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let content = "M 000000000001 vs code=aa\n";
        let sha = git_blob_sha(content.as_bytes());
        let list = format!(
            r#"[{{"name":"PPSA00001_1.kpm","type":"file","sha":"{sha}","download_url":"http://127.0.0.1:{port}/raw/PPSA00001_1.kpm"}}]"#
        );
        let content_owned = content.to_string();
        let server = std::thread::spawn(move || {
            for _ in 0..2 {
                let (mut s, _) = listener.accept().unwrap();
                let mut buf = [0u8; 4096];
                let n = s.read(&mut buf).unwrap();
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                assert!(req.to_ascii_lowercase().contains("user-agent: pipeline-share-server"));
                let body = if req.starts_with("GET /repos/o/r/contents/assets/pipelines?ref=main") {
                    list.clone()
                } else {
                    content_owned.clone()
                };
                let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            }
        });
        let dir = tmp("http");
        let mut c = cfg(&dir, None);
        c.api = format!("http://127.0.0.1:{port}");
        let r = sync_dir(&HttpFetcher, &c, "main", &dir.join("main"), Instant::now() + Duration::from_secs(10)).unwrap();
        assert_eq!(r, Report { downloaded: 1, cached: 0, failed: 0 });
        assert_eq!(fs::read_to_string(dir.join("main/PPSA00001_1.kpm")).unwrap(), content);
        server.join().unwrap();
        let _ = fs::remove_dir_all(&dir);
    }
}
