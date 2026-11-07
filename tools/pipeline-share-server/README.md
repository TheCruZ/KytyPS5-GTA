# pipeline-share-server

Server that shares the KytyPS5 pipeline manifest (`.kpm`, see
`src/graphics/host_gpu/renderer/pipeline/pipelineManifest.cpp` and `assets/pipelines`) between
users. Emulators that opted in upload the records they discover for the first time and download
the ones they lack, so what some play benefits the others in real time. A record holds only
hashes and pipeline state, never game code.

The server keeps everything in memory, saves it to disk every few minutes and on exit, and
exports each partition as `.kpm` files ready to merge into `assets/pipelines`.

## Seeds: the manifests of the repository

At every start, before listening, the server downloads the `.kpm` files of the repository
(`assets/pipelines` on `main` of `TheCruZ/KytyPS5-GTA`, through the public GitHub API; a file is
only downloaded when its Git blob id changed and its content is checked against it) and keeps a
copy in `<data-dir>/seeds/main/`. Their records are **published** (given a sequence number, in
file order, so modules come before the pipelines that name them) unless already published: the
union of what the players sent and what the repository ships, nothing is ever removed. A record
of the repository that a player uploads is not counted as new, and the export leaves out
everything the repository has (the export is what is new and must go into the repository).
If GitHub cannot be reached (no network, rate limit, 60 s in total) the copy on disk is used,
the log says so and the server starts anyway.

Which clients get what:

- **New clients** (`proto` 2): with a cursor they get every published record after it. A client
  that has just been installed, updated to another release or never reconciled sends, in each
  page of its first round, `have`: the hashes of the records of its `assets/pipelines` file and
  its shared manifest. It then only gets what it does not have (records of other players, and
  what a newer repository ships). When it ends, it goes back to the cursor alone.
- **The first clients** (v19, no `proto` and no `have`): they get what the *default release*
  does not ship (`--default-release v19`). That release is downloaded once from the tag of the
  same name of the repository (immutable; `<data-dir>/seeds/v19/`, marked `.complete`, never asked
  again). Without it they would be sent what they already have.

Flow of a new release: push to `main`, **restart the server before publishing the release** (it
downloads the new manifests and publishes their records), and then publish. The clients of the
new release reconcile with their manifests and receive only what the server has beyond them;
the clients of the old release (and v19) receive what the new release brings through the cursor.
Nothing is copied by hand.

## Home page

`GET /` is a page for people (static HTML with its style inline: no scripts, nothing from other
sites, light and dark, narrow screens): what the server is, links to the repository and its
releases, and a table with one row per game and version (grouped by game, by name) with the
partition's own records and a total row:

- **Shaders**: distinct shaders (`code=`) of its own published modules; **pipelines**: its own
  published pipelines. Each split into those the latest release ships and those *from players*
  (what players sent since, and what the repository gained after the release).
- **From other regions and versions**: what it holds as copies (shaders none of its own modules has,
  and pipelines).

Game names come from `src/titles.txt` (a copy of `_Tools/pipelines/titles.txt`), then from the
regions table; a title without a name shows alone. The test partitions (`TEST*`) are left out.

The latest release is asked at every start (`GET /repos/<repo>/releases/latest` of the GitHub API)
and its manifests are downloaded once, as the default release (`seeds/<tag>/`, marked `.complete`,
never asked again); they are only compared with, never published, so what clients get does not
change. Without GitHub the newest complete copy in `seeds/` is used; with none, the seeds of `main`,
and the page says so. `--latest-release <tag>` names it instead (a folder of `--known` of that name
wins). The page says which release and the time of its figures.

The page is made at most once every 60 s (the requests in between get the same HTML) and has a
limit of its own of 120 requests per IP every 10 minutes, apart from the one of `/v1/sync`. It is
served as `text/html; charset=utf-8` with a `Content-Security-Policy` that only allows its inline
style; every text in it is escaped.

## One game in several regions and close versions

A partition is one title and one version, but the regions of a game (GTA V is PPSA04263 in Europe
and PPSA04264 in America) run the same shaders, and so do close versions of a title. The server
shares the records of the partitions of one game, so that a player of one region or version gets
what the players of the others recorded. It is safe: a module record is only used if the game maps
a shader with its `code=` (the hash of the shader binary), a pipeline only by a draw whose key is
exactly the same; what is never used only costs some work when the manifest is loaded.

- **One game**: the titles of one line of the regions table (`src/regions.txt`, built in; `--regions
  <file>` replaces it: `<name>: <title> <title> ...`, one game per line, `#` starts a comment), or
  the versions of a title that is in no line. Only partitions of the same format.
- **Which partitions share** (a *link*): two regions of one version, always. Two versions, when at
  least 50 % (`--merge-share`) of the shaders of the published modules of the one with fewer
  shaders are in the other, and it has at least 100 (`--merge-min-shaders`; fewer, such as the
  menus alone, use libraries that unrelated versions have in common). Only the records the partition
  has as its own count (not its copies). So GTA V PPSA04264 01.000 and 01.005 (1-4 % in common with
  the others) stay apart, while 01.010 and 01.011 of both regions share (80-99 %), and the four
  partitions of ASTRO BOT too (53-99 %). The links are reviewed at start, after the seeds, and
  whenever the shaders of a partition grow. **A link stays** (it is kept in `state.json`): once the
  records are shared, the players of a partition no longer send what they received and the share
  would seem to fall.
- **Copies**: when a record is published as its own in a partition (a player sent it with quorum,
  or its release ships it), the server publishes a copy in every linked partition that has no record
  of that line (or, for a module, of its id), after the modules a pipeline names. A copy gets the
  next sequence number of the partition, so the cursors of its clients are not touched, and is
  served like any record (also to the v19 clients: their release does not ship it). Only own
  records are copied (a copy is not copied on), pipelines are checked again in the destination
  (`validate::pipeline_link_error`, with its modules) and tessellation records are not copied (the
  emulator does not take them from the shared manifest). A copy never takes the place of a
  tombstone, and a pipeline the destination removed is not copied again (it does not link there).
  At start, every linked pair gets what it lacks, in key order (the result depends only on the
  state, not on the order the records arrived in). The first request of a client of a region whose
  version another region has creates its partition and gets the records at once, even if it sends
  nothing.
- **A copy is not a client**: it needs no quorum in the destination (the source had it) and adds
  no pair. When the clients of the partition itself send a copy (with quorum), it becomes its own
  record, in its place; a record that was pending there keeps its clients. A copied module that an
  own pipeline names becomes its own too: its players recorded it (the id of a module is the hash of
  its line) but did not send it, as they had it, and the export of the partition needs it.
- **Limits**: copies have a limit of their own, `--partition-cap` again, so they never take the room
  of the records of the partition.
- **Export**: only the own records, as before: the repository stays one manifest per title and
  version, and the copies are in the manifest they come from.
- `--no-merge` makes no links nor copies (those already made are still served).

At start the log has one line per game with more than one partition: the pairs linked and apart
with the shaders in common of each (`common of fewer`), and the copies its partitions hold:

```
merge GTA V (v1): together PPSA04263_01.010.002 + PPSA04263_01.011.001 91.6% (1283 of 1400), ...; apart PPSA04264_01.000.000 + PPSA04264_01.005.000 4.2% (23 of 548), ...; copies 11797 (11797 new)
```

The `sync` lines say how many copies a request made (`copied N`).

## Build and run

```
cargo build --release
./target/release/pipeline-share-server --data-dir data --listen 127.0.0.1:8787 --quorum 2
```

| Option | Default | Meaning |
|---|---|---|
| `--listen` | `127.0.0.1:8787` | HTTP address (HTTPS goes through a reverse proxy) |
| `--data-dir` | `data` | `state.json`, `export/` and `seeds/` |
| `--seed-repo` / `--seed-ref` / `--seed-path` | `TheCruZ/KytyPS5-GTA` / `main` / `assets/pipelines` | where the seeds are downloaded from at every start |
| `--seed-api` / `--seed-timeout` | `https://api.github.com` / 60 | GitHub API and seconds for all the downloads of a start (then the copies on disk are used) |
| `--no-seed` | off | no download and no copies from `seeds/` (tests, offline) |
| `--default-release` | `default` | release whose records are left out for clients that send neither `proto` nor `have`. If `--known` has no release with that name and it is not `default`, it is downloaded once from the tag of that name (`v19`) |
| `--known <dir>` | none | optional, for tests or without network: every **subfolder** is a release with its `.kpm` files (`<TITLE>_<VERSION>.kpm`, as `assets/pipelines`): `known/v19/`, `known/v20/`. `.kpm` files directly in the folder are the release `default`. Same treatment as the seeds |
| `--quorum` | 2 | distinct clients from distinct IPs needed to publish a record (1 publishes at once) |
| `--save-interval` | 300 | seconds between saves (also on Ctrl-C / SIGTERM) |
| `--real-ip-header` | none | header with the real client IP set by the proxy (`X-Real-IP`). When set it is required (400 without a valid IP) and replaces the TCP peer for quorum, limits and the IP hash. Only use it behind a proxy that overwrites it, with the port not reachable from outside |
| `--ip-requests` / `--ip-window` | 30 / 600 | requests per IP per window (seconds) |
| `--client-daily` / `--ip-daily` | 20000 / 100000 | new records per client / per IP per UTC day |
| `--partition-cap` | 200000 | records (published and pending) per partition, and as many copies |
| `--max-partitions` | 500 | partitions (title and version) accepted |
| `--regions` | built in (`src/regions.txt`) | file with the titles of one game in several regions |
| `--merge-share` / `--merge-min-shaders` | 50 / 100 | two versions share their records when this % of the shaders of the one with fewer are in the other, and it has at least this many |
| `--no-merge` | off | no copies between the partitions of one game |
| `--latest-release` | the latest release on GitHub | release the home page compares its figures with |

`RUST_LOG=debug` logs every request (by default only the ones that publish something, saves and
errors). IPs are only kept as salted hashes (the salt lives in `state.json`); client ids too.

## Output

`data/state.json` (written to a temporary file and renamed, loaded at start) and, per format and
partition, in `data/export/v1/` (one folder per format version):

- `<TITLE>_<VERSION>.kpm`: published records in publication order (modules before the
  pipelines that name them), without what the seeds (the repository) already have and without the
  copies of other regions and versions;
- `<TITLE>_<VERSION>.pending.kpm`: records without quorum yet (or pipelines waiting for modules).

Both start with the format header, so they merge with `assets/pipelines/<TITLE>_<VERSION>.kpm`
by appending the lines that are not there.

## Protocol v1

`GET /` is the home page (see above). `GET /v1/health` answers 200 `ok`.

`POST /v1/sync`, JSON body (`Content-Encoding: gzip` accepted), response JSON gzipped when the
client sends `Accept-Encoding: gzip`:

```json
{"client": "<uuid v4>", "emulator": "<free text, <=64>", "format": "# kyty pipeline manifest v1",
 "title": "PPSA21564", "version": "01.007.000", "cursor": 0, "records": ["M ...", "P ...", "T ..."],
 "proto": 2, "have": ["<16 hex digits>", "..."]}
```

`proto` (2 for the clients that reconcile) and `have` are optional. `have` is only sent while the
client reconciles its manifests with the server: the hashes (first 8 bytes of the SHA-256 of the
record line, in hex) of the records it already has, at most `partition_cap` of them (413 beyond).

Response 200: `{"cursor": <u64>, "records": [...], "more": <bool>}`: the published records of the
partition (format, title, version) after `cursor`, in order. At most 20000 published records are
*reviewed* per response, and the ones the client must not get are left out, so a response can
carry fewer records (even none) with `more` set; the cursor stops at the last record reviewed.
While `more` is true the client repeats with the new cursor (and the same `have`). A cursor
beyond what the server has (it lost its state) restarts from 0.

What is left out: with `have`, the records whose hash is in it; with `proto` 2 and no `have`,
nothing; with neither (the first clients), the records of the default release.

Errors: 400 (invalid request, short text; also when every record is garbage), 413 (body over 4 MiB
decompressed or over 5000 records), 429 with `Retry-After` (IP limit, too many partitions).

Validation: `title` `^[A-Z]{4}[0-9]{5}$`; `version` `^[0-9A-Za-z._-]{1,32}$`; `format` starts with
`# kyty pipeline manifest v` and is at most 64 bytes (what follows must be `[0-9A-Za-z._-]+`, it
names the export folder); `client` a UUID v4; each record starts with `M `, `P ` or `T `, only
ASCII 0x20-0x7E, at most 8192 bytes (an `M` has a 12 hex digit id, the `v=`/`h=`/`d=`/`p=` fields of
a `P` hold ids of that form); invalid records are dropped one by one.

Rules: the identity of a record is its exact line. A record is published (given the next
sequence number and served) once `--quorum` distinct clients from distinct IPs sent it. A `P` is
published only when all the modules it names are published (the repository's count); until then
it waits.

Pipelines that no draw creates from the modules they name are never published, as the emulator's
`LoadManifest()` leaves them out (`validate::pipeline_link_error`, the same checks as
`PipelineLinkError()` in `pipelineManifest.cpp`): a VS or mesh stage, or an LS, HS and TES, with an
optional PS; as many vertex attributes as the VS or LS reads (none for a mesh stage), each from a
binding of the pipeline; the LS's patch size, or patches of 3 points for a rectangle list. The
modules are the published `M` records of the same partition (the first one of an id). Recordings
made before the emulator fixed `RecordPipeline()` mixed up the places of their modules and wrote
such pipelines (Elden Ring Nightreign crashed creating one whose vertex stage was a TES). A
pipeline whose modules are all published is checked when it arrives, and one that waits for them
when they are: it is dropped (not the request) and counted in the `sync` log line (`unlinked N`,
with an example). The seeds are checked as they are published (a `seeds <title>_<version>: left
out N pipelines` warning at start), and at every start the server takes out of its state the
pipelines it published or kept before it checked them (`state <title>_<version>: removed ...`):
a published one leaves an empty line in `state.json` (a tombstone, neither served nor exported)
so that the cursors of the clients keep pointing at the same records. When it removes any, it
copies `state.json` to `state.json.before-unlinked-<unix time>` and saves the cleaned state at
once (`state cleaned: N pipelines removed`).
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
    data/       state.json, export/ and seeds/   (the container runs as uid 65532)
```

The compose file goes next to `src/` and `data/`, and `.env` tells it where the
sources are (`SRC`, default `.` when it is run from the crate folder). First start:

```
mkdir -p /servers/kytypipes/{src,data}
# copy tools/pipeline-share-server/* (not target/) to /servers/kytypipes/src/
cp /servers/kytypipes/src/docker-compose.yml /servers/kytypipes/
echo 'SRC=./src' > /servers/kytypipes/.env
sudo chown 65532:65532 /servers/kytypipes/data
cd /servers/kytypipes
docker compose build
docker compose up -d
docker compose logs -f
```

Update (new server code): copy the new sources over `src/` (and the compose file if it changed), then `docker compose build &&
docker compose up -d`. The container is stopped with SIGTERM and saves its state before exiting
(`stop_grace_period` can be raised for a big state). New repository manifests: push them to
`main` and `docker compose restart` (at every start the server downloads them from GitHub and
publishes their records; pending records that the repository now ships become published, the
published ones stay in the state, and the export leaves out whatever the repository has). The
container needs to reach `api.github.com` and `raw.githubusercontent.com` over HTTPS (the
certificates are embedded in the binary); without them it starts with the copy of `data/seeds/`.
Restart it before publishing the release whose manifests changed.

The state keeps format version 1: the records of the repository are published as any other and
are told apart at run time (the files in `seeds/`), so a state written by an older server loads
unchanged. Copies are saved as tombstones in their places of `published` plus a `copies` list per
partition (`at`, `line` and the clients that sent it), and the links in `links`: a server without
copies loads such a state, serves and exports none of the copies, and its cursors still hold; when
the newer one starts again on a state that lost them, it copies them again after the end.

Update that brings the copies between regions and versions (no change in the compose file):
copy the new sources over `src/`, then `docker compose build && docker compose up -d`, and check
in `docker compose logs --since 5m` the `regions: 4 games (...)` line, one `merge <game> (v1): ...`
line per game (GTA V and ASTRO BOT `together` with thousands of copies, Demon's Souls and GTA: San
Andreas `apart`), `seeds main: ... 0 failed` and `listening on`. The first save after it (within
`--save-interval`) makes `state.json` larger (about 6 MB more with the state of November 2026). To
go back, the previous image with the same `data/` (it ignores `copies` and `links`).

Update that brings the home page (no change in the compose file nor in nginx, which already
forwards `/`): the same steps; the log has `latest release: v20`, `seeds v20: N downloaded, 0
cached, 0 failed` the first time (`complete copy on disk` after it) and `home page: figures
compared with v20 (/data/seeds/v20)`. Each start asks the GitHub API once more (the latest
release). Check `https://kytypipes.mhcruz.com/` in a browser.

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

`cargo test` (quorum, dependencies of `P`, cursor and `more`, releases and seeds, default-release and
`have` filters, filtered pages, state of an older server, GitHub download with a fake network and
a local HTTP server, validation, pipelines that no draw creates in uploads, seeds and an older
state, limits, HTTP with gzip, persistence save/restart/export; the regions table, copies between
regions of one version and between versions with enough shaders in common, modules before the
pipelines, quorum and promotion of copies, tombstones and limits, export without copies and the
state with and without them; the latest release and its copy on disk, the figures of the home page
(own, in the release, from players, copies), its grouping, escaping, cache and limit) and `cargo
clippy --all-targets`.
