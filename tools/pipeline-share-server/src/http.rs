//! HTTP layer: `GET /v1/health` and `POST /v1/sync`.

use crate::limits::Limits;
use crate::store::{PartKey, Store};
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

pub struct App {
    pub store: Mutex<Store>,
    pub limits: Mutex<Limits>,
    /// Header with the client IP set by the reverse proxy (e.g. `X-Real-IP`). When set, it is
    /// required and replaces the TCP peer address.
    pub real_ip_header: Option<String>,
    pub clock: Box<dyn Fn() -> u64 + Send + Sync>,
}

impl App {
    pub fn new(store: Store, limits: Limits, real_ip_header: Option<String>) -> Self {
        Self { store: Mutex::new(store), limits: Mutex::new(limits), real_ip_header, clock: Box::new(unix_now) }
    }

    pub fn store(&self) -> MutexGuard<'_, Store> {
        self.store.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn limits_gc(&self, now: u64) {
        self.limits().gc(now);
    }

    fn limits(&self) -> MutexGuard<'_, Limits> {
        self.limits.lock().unwrap_or_else(|e| e.into_inner())
    }
}

pub fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

pub fn router(app: Arc<App>) -> Router {
    Router::new()
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
    let (cursor, lines, more) = store.fetch(&key, req.cursor);
    drop(store);
    let level = if outcome.published > 0 { tracing::Level::INFO } else { tracing::Level::DEBUG };
    let msg = format!(
        "sync {}/{} client {:08x} rx {} valid {} new {} published {} dropped {} tx {} more {}",
        req.title,
        req.version,
        client as u32,
        received,
        records.len(),
        outcome.counted,
        outcome.published,
        outcome.dropped,
        lines.len(),
        more
    );
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
    const P1: &str = "P v=000000000001 plain";
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
