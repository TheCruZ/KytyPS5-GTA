//! HTTP layer: `GET /` (the home page), `GET /v1/health` and `POST /v1/sync`.

use crate::home::{self, Page};
use crate::limits::{LimitConfig, Limits};
use crate::store::{self, Filter, PartKey, Store};
use crate::validate::{self, MAX_BODY_BYTES, MAX_RECORDS_PER_REQUEST};
use axum::body::{Body, Bytes};
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Extension, Router};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, Mutex, MutexGuard};
use tower_http::compression::CompressionLayer;

/// Seconds the home page is served from its cache before it is made again.
pub const PAGE_EVERY: u64 = 60;
/// Requests of the home page per IP in each window (seconds), apart from those of `/v1/sync`.
pub const PAGE_REQUESTS: u32 = 120;
pub const PAGE_WINDOW: u64 = 600;

pub struct App {
    pub store: Mutex<Store>,
    pub limits: Mutex<Limits>,
    /// Header with the client IP set by the reverse proxy (e.g. `X-Real-IP`). When set, it is
    /// required and replaces the TCP peer address.
    pub real_ip_header: Option<String>,
    pub clock: Box<dyn Fn() -> u64 + Send + Sync>,
    /// The texts of the home page, its cache (made at, HTML) and its own request limit.
    pub page: Page,
    home: Mutex<Option<(u64, Bytes)>>,
    page_limits: Mutex<Limits>,
}

impl App {
    pub fn new(store: Store, limits: Limits, real_ip_header: Option<String>) -> Self {
        Self {
            store: Mutex::new(store),
            limits: Mutex::new(limits),
            real_ip_header,
            clock: Box::new(unix_now),
            page: Page::default(),
            home: Mutex::new(None),
            page_limits: Mutex::new(Limits::new(LimitConfig {
                ip_requests: PAGE_REQUESTS,
                ip_window_secs: PAGE_WINDOW,
                ..LimitConfig::default()
            })),
        }
    }

    pub fn with_page(mut self, page: Page) -> Self {
        self.page = page;
        self
    }

    pub fn store(&self) -> MutexGuard<'_, Store> {
        self.store.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn limits_gc(&self, now: u64) {
        self.limits().gc(now);
        self.page_limits().gc(now);
    }

    fn limits(&self) -> MutexGuard<'_, Limits> {
        self.limits.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn page_limits(&self) -> MutexGuard<'_, Limits> {
        self.page_limits.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The home page: the cached one while it is younger than `PAGE_EVERY` seconds, else made
    /// again from the store.
    pub fn home_html(&self) -> Bytes {
        let now = (self.clock)();
        let mut cache = self.home.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((at, html)) = &*cache {
            if now >= *at && now < at + PAGE_EVERY {
                return html.clone();
            }
        }
        let stats = self.store().stats();
        let html = Bytes::from(home::render(&self.page, &stats, now));
        *cache = Some((now, html.clone()));
        html
    }
}

pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

pub fn router(app: Arc<App>) -> Router {
    Router::new()
        .route("/", get(home_page).route_layer(middleware::from_fn_with_state(app.clone(), page_rate_limit)))
        .route("/v1/health", get(|| async { "ok" }))
        .route("/v1/sync", post(sync).route_layer(middleware::from_fn_with_state(app.clone(), rate_limit)))
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY_BYTES))
        .layer(CompressionLayer::new().gzip(true))
        .with_state(app)
}

/// Salted hash of the (masked) client address, set by the rate limit for the handler.
#[derive(Clone, Copy)]
struct IpHash(u64);

/// The client address: the proxy's header when configured (`None` when it is missing or not an
/// IP), else the TCP peer.
fn client_ip(app: &App, headers: &HeaderMap, peer: Option<IpAddr>) -> Option<IpAddr> {
    match app.real_ip_header.as_deref() {
        Some(name) => headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            // Several values: the last one is the one our own proxy appended.
            .and_then(|v| v.rsplit(',').next())
            .and_then(|v| v.trim().parse::<IpAddr>().ok()),
        None => Some(peer.unwrap_or(IpAddr::from([0, 0, 0, 0]))),
    }
}

/// IPv6 addresses are masked to their /64 and mapped IPv4 ones are IPv4.
fn mask_ip(ip: IpAddr) -> String {
    match ip {
        IpAddr::V4(v4) => v4.to_string(),
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => v4.to_string(),
            None => {
                let s = v6.segments();
                format!("{:x}:{:x}:{:x}:{:x}", s[0], s[1], s[2], s[3])
            }
        },
    }
}

async fn rate_limit(State(app): State<Arc<App>>, mut req: Request, next: Next) -> Response {
    let peer = req.extensions().get::<ConnectInfo<SocketAddr>>().map(|c| c.0.ip());
    let Some(ip) = client_ip(&app, req.headers(), peer) else {
        return bad("missing or invalid real ip header");
    };
    let ip = mask_ip(ip);
    let hash = app.store().hash("ip", &ip);
    let now = (app.clock)();
    if let Err(retry) = app.limits().check_ip(hash, now) {
        tracing::debug!("429 rate limit for ip hash {hash:016x}");
        return too_many(retry, "too many requests");
    }
    req.extensions_mut().insert(IpHash(hash));
    next.run(req).await
}

/// The home page has a request limit of its own: reading it does not use up what the emulator
/// of that IP may send.
async fn page_rate_limit(State(app): State<Arc<App>>, req: Request, next: Next) -> Response {
    let peer = req.extensions().get::<ConnectInfo<SocketAddr>>().map(|c| c.0.ip());
    let Some(ip) = client_ip(&app, req.headers(), peer) else {
        return bad("missing or invalid real ip header");
    };
    let hash = app.store().hash("ip", &mask_ip(ip));
    let now = (app.clock)();
    if let Err(retry) = app.page_limits().check_ip(hash, now) {
        return too_many(retry, "too many requests");
    }
    next.run(req).await
}

async fn home_page(State(app): State<Arc<App>>) -> Response {
    let html = tokio::task::spawn_blocking(move || app.home_html()).await.unwrap_or_default();
    let mut resp = Response::new(Body::from(html));
    let headers = resp.headers_mut();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8"));
    headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("public, max-age=60"));
    headers.insert(header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    // Inline style only: no scripts, no other resources, no frames.
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static("default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"),
    );
    resp
}

fn too_many(retry_after: u64, text: &'static str) -> Response {
    let mut resp = (StatusCode::TOO_MANY_REQUESTS, text).into_response();
    if let Ok(v) = HeaderValue::from_str(&retry_after.to_string()) {
        resp.headers_mut().insert(header::RETRY_AFTER, v);
    }
    resp
}

fn bad(text: &'static str) -> Response {
    (StatusCode::BAD_REQUEST, text).into_response()
}

#[derive(Deserialize)]
struct SyncRequest {
    client: String,
    emulator: String,
    format: String,
    title: String,
    version: String,
    cursor: u64,
    records: Vec<String>,
    /// Protocol revision of the client. Absent in the first clients (they get what the default
    /// release does not ship); 2 or more answer with everything published unless `have` is set.
    #[serde(default)]
    proto: Option<u32>,
    /// Hashes (first 8 bytes of the SHA-256, 16 hex digits) of the records the client has, sent
    /// while it reconciles its manifests with the server: it only gets what it does not have.
    #[serde(default)]
    have: Option<Vec<String>>,
}

#[derive(Serialize)]
struct SyncResponse<'a> {
    cursor: u64,
    records: &'a [Arc<str>],
    more: bool,
}

fn gunzip(data: &[u8]) -> Result<Vec<u8>, Box<Response>> {
    let mut out = Vec::new();
    let limit = MAX_BODY_BYTES as u64 + 1;
    if GzDecoder::new(data).take(limit).read_to_end(&mut out).is_err() {
        return Err(Box::new(bad("invalid gzip body")));
    }
    if out.len() > MAX_BODY_BYTES {
        return Err(Box::new((StatusCode::PAYLOAD_TOO_LARGE, "body too large").into_response()));
    }
    Ok(out)
}

async fn sync(
    State(app): State<Arc<App>>,
    Extension(IpHash(ip)): Extension<IpHash>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let encoding = headers
        .get(header::CONTENT_ENCODING)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.trim().to_ascii_lowercase());
    let body = match encoding.as_deref() {
        None | Some("") | Some("identity") => body.to_vec(),
        Some("gzip") => match gunzip(&body) {
            Ok(b) => b,
            Err(r) => return *r,
        },
        Some(_) => return bad("unsupported content-encoding"),
    };
    let Ok(req) = serde_json::from_slice::<SyncRequest>(&body) else {
        return bad("invalid json");
    };
    if !validate::valid_client(&req.client) {
        return bad("invalid client");
    }
    if !validate::valid_emulator(&req.emulator) {
        return bad("invalid emulator");
    }
    if !validate::valid_format(&req.format) {
        return bad("invalid format");
    }
    if !validate::valid_title(&req.title) {
        return bad("invalid title");
    }
    if !validate::valid_version(&req.version) {
        return bad("invalid version");
    }
    if req.records.len() > MAX_RECORDS_PER_REQUEST {
        return (StatusCode::PAYLOAD_TOO_LARGE, "too many records").into_response();
    }
    let received = req.records.len();
    let mut records = Vec::with_capacity(received);
    for line in req.records {
        if let Some(parsed) = validate::parse_record(&line) {
            records.push((line, parsed));
        }
    }
    if received > 0 && records.is_empty() {
        return bad("no valid records");
    }

    let have_count = req.have.as_ref().map(Vec::len);
    let cap = app.store().settings().partition_cap;
    let filter = match &req.have {
        Some(have) => {
            if have.len() > cap {
                return (StatusCode::PAYLOAD_TOO_LARGE, "too many hashes").into_response();
            }
            Filter::Have(have.iter().filter_map(|h| store::parse_hash(h)).collect())
        }
        None if req.proto.is_some_and(|p| p >= 2) => Filter::None,
        None => Filter::DefaultRelease,
    };

    let key = PartKey::new(&req.format, &req.title, &req.version);
    let now = (app.clock)();
    let mut store = app.store();
    let client = store.hash("client", &req.client);
    let mut limits = app.limits();
    let budget = limits.remaining_new(client, ip, now);
    let outcome = store.submit(&key, client, ip, &records, budget);
    limits.add_new(client, ip, outcome.counted, now);
    drop(limits);
    if outcome.partition_refused {
        return too_many(3600, "too many partitions");
    }
    let (cursor, lines, more) = store.fetch(&key, req.cursor, &filter);
    drop(store);
    let unlinked = outcome.unlinked.count;
    let level = if outcome.published > 0 || unlinked > 0 || outcome.copied > 0 {
        tracing::Level::INFO
    } else {
        tracing::Level::DEBUG
    };
    let mut msg = format!(
        "sync {}/{} client {:08x} rx {} valid {} new {} published {} dropped {} unlinked {} copied {} have {:?} cursor {}->{} tx {} more {}",
        req.title,
        req.version,
        client as u32,
        received,
        records.len(),
        outcome.counted,
        outcome.published,
        outcome.dropped,
        unlinked,
        outcome.copied,
        have_count,
        req.cursor,
        cursor,
        lines.len(),
        more
    );
    // Pipelines that no draw creates from the modules they name are left out one by one.
    if let Some(example) = &outcome.unlinked.example {
        msg += &format!("; unlinked pipelines such as {example}");
    }
    if level == tracing::Level::INFO {
        tracing::info!("{msg}");
    } else {
        tracing::debug!("{msg}");
    }
    let json = match serde_json::to_vec(&SyncResponse { cursor, records: &lines, more }) {
        Ok(j) => j,
        Err(_) => return (StatusCode::INTERNAL_SERVER_ERROR, "encoding error").into_response(),
    };
    let mut resp = Response::new(Body::from(json));
    resp.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
    resp
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::LimitConfig;
    use crate::store::Settings;
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use http_body_util::BodyExt;
    use std::io::Write;
    use tower::ServiceExt;

    const FMT: &str = "# kyty pipeline manifest v1";
    const M1: &str = "M 000000000001 vs code=aa";
    const P1: &str = "P v=000000000001";
    const CLIENT_A: &str = "0b1c2d3e-4f50-4a6b-8c7d-9e0f1a2b3c4d";
    const CLIENT_B: &str = "1b1c2d3e-4f50-4a6b-8c7d-9e0f1a2b3c4d";

    fn app(quorum: usize, ip_requests: u32) -> Arc<App> {
        let store = Store::with_salt(Settings { quorum, ..Default::default() }, [1; 16]);
        let limits = Limits::new(LimitConfig { ip_requests, ..Default::default() });
        Arc::new(App::new(store, limits, Some("x-real-ip".into())))
    }

    fn body(client: &str, title: &str, cursor: u64, records: &[&str]) -> String {
        serde_json::json!({
            "client": client, "emulator": "kyty test", "format": FMT, "title": title,
            "version": "01.007.000", "cursor": cursor, "records": records,
        })
        .to_string()
    }

    async fn call(
        app: &Arc<App>,
        ip: &str,
        body: Vec<u8>,
        gzip_in: bool,
        gzip_out: bool,
    ) -> (StatusCode, HeaderMap, Vec<u8>) {
        let mut b = Request::builder().method("POST").uri("/v1/sync").header("x-real-ip", ip);
        if gzip_in {
            b = b.header("content-encoding", "gzip");
        }
        if gzip_out {
            b = b.header("accept-encoding", "gzip");
        }
        let resp = router(app.clone()).oneshot(b.body(Body::from(body)).unwrap()).await.unwrap();
        let (parts, body) = resp.into_parts();
        let bytes = body.collect().await.unwrap().to_bytes().to_vec();
        (parts.status, parts.headers, bytes)
    }

    fn gz(data: &[u8]) -> Vec<u8> {
        let mut e = GzEncoder::new(Vec::new(), Compression::default());
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }

    fn json(bytes: &[u8]) -> serde_json::Value {
        serde_json::from_slice(bytes).unwrap()
    }

    #[tokio::test]
    async fn health() {
        let resp = router(app(1, 10))
            .oneshot(Request::get("/v1/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&bytes[..], b"ok");
    }

    #[tokio::test]
    async fn two_clients_share_through_quorum_with_gzip() {
        let app = app(2, 10);
        // Client A uploads gzip and gets nothing: no quorum yet.
        let (st, _, out) = call(&app, "1.1.1.1", gz(body(CLIENT_A, "PPSA21564", 0, &[M1, P1]).as_bytes()), true, false).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(json(&out), serde_json::json!({"cursor": 0, "records": [], "more": false}));
        // Client B from another IP confirms them and receives them, gzipped.
        let (st, headers, out) = call(&app, "2.2.2.2", body(CLIENT_B, "PPSA21564", 0, &[M1, P1]).into_bytes(), false, true).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(headers.get("content-encoding").unwrap(), "gzip");
        let mut text = String::new();
        GzDecoder::new(&out[..]).read_to_string(&mut text).unwrap();
        assert_eq!(json(text.as_bytes()), serde_json::json!({"cursor": 2, "records": [M1, P1], "more": false}));
        // Client A polls with its cursor and with an empty list.
        let (_, _, out) = call(&app, "1.1.1.1", body(CLIENT_A, "PPSA21564", 0, &[]).into_bytes(), false, false).await;
        assert_eq!(json(&out)["records"].as_array().unwrap().len(), 2);
        let (_, _, out) = call(&app, "1.1.1.1", body(CLIENT_A, "PPSA21564", 2, &[]).into_bytes(), false, false).await;
        assert_eq!(json(&out), serde_json::json!({"cursor": 2, "records": [], "more": false}));
        // Another partition is empty.
        let (_, _, out) = call(&app, "1.1.1.1", body(CLIENT_A, "PPSA21565", 0, &[]).into_bytes(), false, false).await;
        assert_eq!(json(&out)["records"].as_array().unwrap().len(), 0);
    }

    const M2: &str = "M 000000000002 ps code=bb";

    fn seeded_app() -> Arc<App> {
        use crate::store::{Known, Release};
        let a = app(1, 100);
        let key = PartKey::new(FMT, "PPSA21564", "01.007.000");
        let mut v19 = Known::default();
        v19.add(M1);
        let mut v20 = Known::default();
        v20.add(M1);
        v20.add(M2);
        let release = |name: &str, k: Known| Release {
            name: name.into(),
            parts: std::collections::HashMap::from([(key.clone(), k)]),
        };
        {
            let mut store = a.store();
            store.settings_mut().default_release = "v19".into();
            store.set_known(vec![release("v19", v19), release("v20", v20)]);
        }
        a
    }

    fn records_of(out: &[u8]) -> Vec<String> {
        json(out)["records"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect()
    }

    fn with(extra: serde_json::Value) -> Vec<u8> {
        let mut v: serde_json::Value = serde_json::from_str(&body(CLIENT_A, "PPSA21564", 0, &[])).unwrap();
        for (k, x) in extra.as_object().unwrap() {
            v[k] = x.clone();
        }
        v.to_string().into_bytes()
    }

    #[tokio::test]
    async fn clients_without_proto_do_not_get_the_default_release() {
        let app = seeded_app();
        // The first clients send neither `proto` nor `have`: what v19 ships is left out.
        let (st, _, out) = call(&app, "1.1.1.1", body(CLIENT_A, "PPSA21564", 0, &[]).into_bytes(), false, false).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(records_of(&out), [M2]);
        assert_eq!(json(&out)["cursor"], 2);
    }

    #[tokio::test]
    async fn new_clients_get_everything_or_what_they_lack() {
        let app = seeded_app();
        // proto 2 without `have`: normal mode, no filter.
        let (_, _, out) = call(&app, "1.1.1.1", with(serde_json::json!({"proto": 2})), false, false).await;
        assert_eq!(records_of(&out), [M1, M2]);
        // With `have` (reconciliation): only what is not in it; an empty list gets everything.
        let have = format!("{:016x}", crate::store::record_hash(M1));
        let (_, _, out) = call(&app, "1.1.1.1", with(serde_json::json!({"proto": 2, "have": [have, "zz", "12"]})), false, false).await;
        assert_eq!(records_of(&out), [M2]);
        let (_, _, out) = call(&app, "1.1.1.1", with(serde_json::json!({"proto": 2, "have": []})), false, false).await;
        assert_eq!(records_of(&out), [M1, M2]);
        // `have` also wins over the missing proto.
        let (_, _, out) = call(&app, "1.1.1.1", with(serde_json::json!({"have": [have]})), false, false).await;
        assert_eq!(records_of(&out), [M2]);
        // More hashes than a partition may hold: 413.
        let many = vec!["0000000000000000"; 200_001];
        let (st, _, _) = call(&app, "1.1.1.1", with(serde_json::json!({"proto": 2, "have": many})), false, false).await;
        assert_eq!(st, StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn the_other_region_gets_the_records_of_its_version() {
        use crate::regions::Regions;
        use crate::store::Merge;
        let regions = Regions::parse("ASTRO BOT: PPSA21564 PPSA21567").unwrap();
        let merge = Merge { regions, share_percent: 50, min_codes: 100 };
        let store = Store::with_salt(Settings { quorum: 1, merge: Some(merge), ..Default::default() }, [1; 16]);
        let limits = Limits::new(LimitConfig { ip_requests: 100, ..Default::default() });
        let app = Arc::new(App::new(store, limits, Some("x-real-ip".into())));
        call(&app, "1.1.1.1", body(CLIENT_A, "PPSA21564", 0, &[M1, P1]).into_bytes(), false, false).await;
        // A client of the other region that has nothing to send gets them at once.
        let (st, _, out) = call(&app, "2.2.2.2", body(CLIENT_B, "PPSA21567", 0, &[]).into_bytes(), false, false).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(json(&out), serde_json::json!({"cursor": 2, "records": [M1, P1], "more": false}));
        // What it sends goes the other way.
        call(&app, "2.2.2.2", body(CLIENT_B, "PPSA21567", 2, &[M2]).into_bytes(), false, false).await;
        let (_, _, out) = call(&app, "1.1.1.1", body(CLIENT_A, "PPSA21564", 2, &[]).into_bytes(), false, false).await;
        assert_eq!(records_of(&out), [M2]);
        // Another version of the same title, with a handful of shaders: apart.
        let other = serde_json::json!({
            "client": CLIENT_B, "emulator": "kyty test", "format": FMT, "title": "PPSA21564",
            "version": "01.018.000", "cursor": 0, "records": ["M 000000000009 ps code=cc"],
        });
        let (_, _, out) = call(&app, "2.2.2.2", other.to_string().into_bytes(), false, false).await;
        assert_eq!(records_of(&out), ["M 000000000009 ps code=cc"]);
    }

    async fn get_home(app: &Arc<App>, ip: &str) -> (StatusCode, HeaderMap, String) {
        let req = Request::get("/").header("x-real-ip", ip).body(Body::empty()).unwrap();
        let resp = router(app.clone()).oneshot(req).await.unwrap();
        let (parts, body) = resp.into_parts();
        let bytes = body.collect().await.unwrap().to_bytes().to_vec();
        (parts.status, parts.headers, String::from_utf8(bytes).unwrap())
    }

    #[tokio::test]
    async fn home_page_is_cached_and_has_its_own_limit() {
        use std::sync::atomic::{AtomicU64, Ordering};
        let time = Arc::new(AtomicU64::new(1_793_909_675));
        let mut a = App::new(
            Store::with_salt(Settings { quorum: 1, ..Default::default() }, [1; 16]),
            Limits::new(LimitConfig { ip_requests: 2, ..Default::default() }),
            Some("x-real-ip".into()),
        );
        let t = time.clone();
        a.clock = Box::new(move || t.load(Ordering::Relaxed));
        let app = Arc::new(a);
        let (st, headers, html) = get_home(&app, "1.1.1.1").await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(headers.get("content-type").unwrap(), "text/html; charset=utf-8");
        assert!(headers.get("content-security-policy").unwrap().to_str().unwrap().starts_with("default-src 'none'"));
        assert!(html.starts_with("<!DOCTYPE html>") && html.contains("Nothing recorded yet."));
        // Reading the page does not use up the requests of /v1/sync (two for this IP).
        for _ in 0..5 {
            assert_eq!(get_home(&app, "1.1.1.1").await.0, StatusCode::OK);
        }
        let (st, _, _) = call(&app, "1.1.1.1", body(CLIENT_A, "PPSA21564", 0, &[M1]).into_bytes(), false, false).await;
        assert_eq!(st, StatusCode::OK);
        // Made at most once a minute: the record shows after 60 s.
        assert!(get_home(&app, "1.1.1.1").await.2.contains("Nothing recorded yet."));
        time.fetch_add(PAGE_EVERY, Ordering::Relaxed);
        let html = get_home(&app, "1.1.1.1").await.2;
        assert!(html.contains("<th colspan=\"10\" scope=\"colgroup\">ASTRO BOT</th>"), "{html}");
        assert!(html.contains("<td class=\"l\">PPSA21564</td><td class=\"l\">01.007.000</td><td>1</td><td>0</td><td>1</td>"));
        // Its own limit.
        for _ in 0..PAGE_REQUESTS - 8 {
            get_home(&app, "1.1.1.1").await;
        }
        let (st, headers, _) = get_home(&app, "1.1.1.1").await;
        assert_eq!(st, StatusCode::TOO_MANY_REQUESTS);
        assert!(headers.get("retry-after").is_some());
        assert_eq!(get_home(&app, "2.2.2.2").await.0, StatusCode::OK);
        // Without the proxy's header: refused, as /v1/sync.
        let req = Request::get("/").body(Body::empty()).unwrap();
        assert_eq!(router(app.clone()).oneshot(req).await.unwrap().status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn invalid_requests() {
        let app = app(1, 100);
        let ok = |t: &str| body(CLIENT_A, t, 0, &[M1]).into_bytes();
        assert_eq!(call(&app, "1.1.1.1", ok("ppsa21564"), false, false).await.0, StatusCode::BAD_REQUEST);
        assert_eq!(call(&app, "1.1.1.1", b"{".to_vec(), false, false).await.0, StatusCode::BAD_REQUEST);
        assert_eq!(call(&app, "1.1.1.1", body("nope", "PPSA21564", 0, &[M1]).into_bytes(), false, false).await.0, StatusCode::BAD_REQUEST);
        assert_eq!(call(&app, "1.1.1.1", b"not gzip".to_vec(), true, false).await.0, StatusCode::BAD_REQUEST);
        // Everything is garbage: 400. Some garbage among good records: dropped one by one.
        assert_eq!(call(&app, "1.1.1.1", body(CLIENT_A, "PPSA21564", 0, &["X 1", "M zz"]).into_bytes(), false, false).await.0, StatusCode::BAD_REQUEST);
        let (st, _, out) = call(&app, "1.1.1.1", body(CLIENT_A, "PPSA21564", 0, &["X 1", M1]).into_bytes(), false, false).await;
        assert_eq!(st, StatusCode::OK);
        assert_eq!(json(&out)["records"], serde_json::json!([M1]));
    }

    #[tokio::test]
    async fn too_large() {
        let app = app(1, 100);
        let many = vec![M1; MAX_RECORDS_PER_REQUEST + 1];
        let (st, _, _) = call(&app, "1.1.1.1", body(CLIENT_A, "PPSA21564", 0, &many).into_bytes(), false, false).await;
        assert_eq!(st, StatusCode::PAYLOAD_TOO_LARGE);
        let big = vec![b' '; MAX_BODY_BYTES + 1];
        let (st, _, _) = call(&app, "1.1.1.1", big.clone(), false, false).await;
        assert_eq!(st, StatusCode::PAYLOAD_TOO_LARGE);
        // A small gzip that inflates beyond the limit.
        let (st, _, _) = call(&app, "1.1.1.1", gz(&big), true, false).await;
        assert_eq!(st, StatusCode::PAYLOAD_TOO_LARGE);
    }

    #[tokio::test]
    async fn rate_limit_by_ip() {
        let app = app(1, 2);
        for _ in 0..2 {
            let (st, _, _) = call(&app, "9.9.9.9", body(CLIENT_A, "PPSA21564", 0, &[]).into_bytes(), false, false).await;
            assert_eq!(st, StatusCode::OK);
        }
        let (st, headers, _) = call(&app, "9.9.9.9", body(CLIENT_A, "PPSA21564", 0, &[]).into_bytes(), false, false).await;
        assert_eq!(st, StatusCode::TOO_MANY_REQUESTS);
        assert!(headers.get("retry-after").unwrap().to_str().unwrap().parse::<u64>().unwrap() > 0);
        // Another IP is not affected.
        let (st, _, _) = call(&app, "8.8.8.8", body(CLIENT_A, "PPSA21564", 0, &[]).into_bytes(), false, false).await;
        assert_eq!(st, StatusCode::OK);
    }

    #[tokio::test]
    async fn real_ip_header_is_required_when_configured() {
        let app = app(1, 10);
        let req = Request::builder()
            .method("POST")
            .uri("/v1/sync")
            .body(Body::from(body(CLIENT_A, "PPSA21564", 0, &[])))
            .unwrap();
        let resp = router(app.clone()).oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        assert_eq!(call(&app, "not-an-ip", body(CLIENT_A, "PPSA21564", 0, &[]).into_bytes(), false, false).await.0, StatusCode::BAD_REQUEST);
    }

    #[test]
    fn ip_masking() {
        assert_eq!(mask_ip("1.2.3.4".parse().unwrap()), "1.2.3.4");
        assert_eq!(mask_ip("::ffff:1.2.3.4".parse().unwrap()), "1.2.3.4");
        assert_eq!(mask_ip("2001:db8:1:2:3:4:5:6".parse().unwrap()), mask_ip("2001:db8:1:2::9".parse().unwrap()));
    }
}
