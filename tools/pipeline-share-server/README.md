# pipeline-share-server

Server that shares the KytyPS5 pipeline manifest (`.kpm`, see
`src/graphics/host_gpu/renderer/pipeline/pipelineManifest.cpp` and `assets/pipelines`) between
users. Emulators that opted in upload the records they discover for the first time and download
the ones they lack, so what some play benefits the others in real time. A record holds only
hashes and pipeline state, never game code.

The server keeps everything in memory, saves it to disk every few minutes and on exit, and
exports each partition as `.kpm` files ready to merge into `assets/pipelines`.

## Build and run

```
cargo build --release
./target/release/pipeline-share-server \
    --known ../../assets/pipelines --data-dir data --listen 127.0.0.1:8787 --quorum 2
```

| Option | Default | Meaning |
|---|---|---|
| `--listen` | `127.0.0.1:8787` | HTTP address (HTTPS goes through a reverse proxy) |
| `--data-dir` | `data` | `state.json` and `export/` |
| `--known <dir>` | none | `.kpm` files of the repository (`<TITLE>_<VERSION>.kpm`): already known, never served nor counted as new, and their modules satisfy the pipelines |
| `--quorum` | 2 | distinct clients from distinct IPs needed to publish a record (1 publishes at once) |
| `--save-interval` | 300 | seconds between saves (also on Ctrl-C / SIGTERM) |
| `--real-ip-header` | none | header with the real client IP set by the proxy (`X-Real-IP`). When set it is required (400 without a valid IP) and replaces the TCP peer for quorum, limits and the IP hash. Only use it behind a proxy that overwrites it, with the port not reachable from outside |
| `--ip-requests` / `--ip-window` | 30 / 600 | requests per IP per window (seconds) |
| `--client-daily` / `--ip-daily` | 20000 / 100000 | new records per client / per IP per UTC day |
| `--partition-cap` | 200000 | records (published and pending) per partition |
| `--max-partitions` | 500 | partitions (title and version) accepted |

`RUST_LOG=debug` logs every request (by default only the ones that publish something, saves and
errors). IPs are only kept as salted hashes (the salt lives in `state.json`); client ids too.

## Output

`data/state.json` (written to a temporary file and renamed, loaded at start) and, per format and
partition, in `data/export/v1/` (one folder per format version):

- `<TITLE>_<VERSION>.kpm`: published records in publication order (modules before the
  pipelines that name them), without what `--known` already has;
- `<TITLE>_<VERSION>.pending.kpm`: records without quorum yet (or pipelines waiting for modules).

Both start with the format header, so they merge with `assets/pipelines/<TITLE>_<VERSION>.kpm`
by appending the lines that are not there.

## Protocol v1

`GET /v1/health` answers 200 `ok`.

`POST /v1/sync`, JSON body (`Content-Encoding: gzip` accepted), response JSON gzipped when the
client sends `Accept-Encoding: gzip`:

```json
{"client": "<uuid v4>", "emulator": "<free text, <=64>", "format": "# kyty pipeline manifest v1",
 "title": "PPSA21564", "version": "01.007.000", "cursor": 0, "records": ["M ...", "P ...", "T ..."]}
```

Response 200: `{"cursor": <u64>, "records": [...], "more": <bool>}`: the published records of the
partition (format, title, version) with sequence number above `cursor`, in order, at most 20000
per response. While `more` is true the client repeats with the new cursor and no records. A
cursor beyond what the server has (it lost its state) restarts from 0.

Errors: 400 (invalid request, short text; also when every record is garbage), 413 (body over 4 MiB
decompressed or over 5000 records), 429 with `Retry-After` (IP limit, too many partitions).

Validation: `title` `^[A-Z]{4}[0-9]{5}$`; `version` `^[0-9A-Za-z._-]{1,32}$`; `format` starts with
`# kyty pipeline manifest v` and is at most 64 bytes (what follows must be `[0-9A-Za-z._-]+`, it
names the export folder); `client` a UUID v4; each record starts with `M `, `P ` or `T `, only
ASCII 0x20-0x7E, at most 8192 bytes (an `M` has a 12 hex digit id, the `v=`/`h=`/`d=`/`p=` fields of
a `P` hold ids of that form); invalid records are dropped one by one.

Rules: the identity of a record is its exact line. A record is published (given the next
sequence number and served) once `--quorum` distinct clients from distinct IPs sent it. A `P` is
published only when all the modules it names are published or known; until then it waits.
When a client has used up its daily budget the rest of its records are ignored (the response
still carries what it lacks); when the IP budget of requests is used up the answer is 429.

## Deployment (Docker, behind Cloudflare and nginx)

Public URL: `https://kytypipes.mhcruz.com`. Cloudflare ends HTTPS, the nginx container
(`nginx-nginx-1`, network `nginx-net`) only accepts Cloudflare IPs and forwards over HTTP to
`kytypipes-app:8787`. The server does not see the real IP as the TCP peer, hence
`--real-ip-header X-Real-IP` (nginx sets it from `CF-Connecting-IP`). The port is not published
on the host.

Folder on the VPS (Ubuntu 24.04, x86_64, Docker; no Rust needed):

```
/servers/kytypipes/
    src/        this crate (Cargo.toml, Cargo.lock, src/, Dockerfile, docker-compose.yml)
    data/       state.json and export/   (the container runs as uid 65532)
    known/      copy of assets/pipelines/*.kpm
```

The compose file goes next to `src/`, `data/` and `known/`, and `.env` tells it where the
sources are (`SRC`, default `.` when it is run from the crate folder). First start:

```
mkdir -p /servers/kytypipes/{src,data,known}
# copy tools/pipeline-share-server/* (not target/) to /servers/kytypipes/src/
cp /servers/kytypipes/src/docker-compose.yml /servers/kytypipes/
echo 'SRC=./src' > /servers/kytypipes/.env
# copy assets/pipelines/*.kpm to /servers/kytypipes/known/
sudo chown 65532:65532 /servers/kytypipes/data
cd /servers/kytypipes
docker compose build
docker compose up -d
docker compose logs -f
```

Update (new server code): copy the new sources over `src/` (and the compose file if it changed), then `docker compose build &&
docker compose up -d`. The container is stopped with SIGTERM and saves its state before exiting
(`stop_grace_period` can be raised for a big state). New repository manifests: copy them to
`known/` and `docker compose restart` (pending records that are now known are dropped,
published ones stay in the state; the export leaves out what is known).

Collecting the result: `data/export/v1/*.kpm` are merged into `assets/pipelines` in the repository.

nginx, `/servers/nginx/conf/kytypipes.conf` (same pattern as the other sites of the VPS; reload
nginx afterwards):

```
server {
    listen 80;
    server_name kytypipes.mhcruz.com;
    include /cfips.conf;
    location / {
        resolver 127.0.0.11 ipv6=off;
        set $upstream kytypipes-app;
        proxy_pass http://$upstream:8787;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-For $http_cf_connecting_ip;
        proxy_set_header X-Forwarded-Proto https;
        client_max_body_size 5M;
    }
}
```

Cloudflare: bot mode or security challenges can block requests that do not come from a browser
(the emulator is not one). Exclude `kytypipes.mhcruz.com/v1/*` from challenges (WAF custom rule
with action Skip, or turn off Bot Fight Mode for that host). Test with
`curl https://kytypipes.mhcruz.com/v1/health`, which must answer `ok`.

## Other proxies (Caddy)

Without Cloudflare, Caddy gives HTTPS with a one-line config. `X-Real-IP` is set by Caddy from the
connection, replacing any value the client sends:

```
kytypipes.example.com {
    reverse_proxy 127.0.0.1:8787 {
        header_up X-Real-IP {remote_host}
    }
}
```

Run the server with `--real-ip-header X-Real-IP`.

## Tests

`cargo test` (quorum, dependencies of `P`, cursor and `more`, known records, validation, limits,
HTTP with gzip, persistence save/restart/export) and `cargo clippy --all-targets`.
