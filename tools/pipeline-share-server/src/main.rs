use clap::Parser;
use pipeline_share_server::http::{self, App};
use pipeline_share_server::limits::{LimitConfig, Limits};
use pipeline_share_server::seed;
use pipeline_share_server::store::{self, Release, Settings, Store};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// Shares the KytyPS5 pipeline manifest between users (protocol v1).
#[derive(Parser, Debug)]
#[command(version)]
struct Args {
    /// Address to listen on (HTTP; put a reverse proxy in front for HTTPS).
    #[arg(long, default_value = "127.0.0.1:8787")]
    listen: SocketAddr,
    /// Directory for `state.json` and `export/`.
    #[arg(long, default_value = "data")]
    data_dir: PathBuf,
    /// Directory with the releases of the repository: one subfolder per release with its
    /// `assets/pipelines` `.kpm` files (`known/v19/`, `known/v20/`). Their records are published
    /// as seeds (never counted as new nor exported); `.kpm` files directly in it form the
    /// release `default`.
    #[arg(long)]
    known: Option<PathBuf>,
    /// Release whose records are left out for the clients that do not say what they have (the
    /// ones published before the reconciliation existed). If `--known` has no release with this
    /// name, it is downloaded once from the tag of that name of the seed repository.
    #[arg(long, default_value = store::LOOSE_RELEASE)]
    default_release: String,
    /// GitHub repository (`owner/name`) whose manifests are downloaded at every start.
    #[arg(long, default_value = "TheCruZ/KytyPS5-GTA")]
    seed_repo: String,
    /// Branch of the seed repository.
    #[arg(long, default_value = "main")]
    seed_ref: String,
    /// Folder of the seed repository with the `.kpm` files.
    #[arg(long, default_value = "assets/pipelines")]
    seed_path: String,
    /// GitHub API address.
    #[arg(long, default_value = "https://api.github.com")]
    seed_api: String,
    /// Seconds for all the downloads of a start; after that the copies on disk are used.
    #[arg(long, default_value_t = 60)]
    seed_timeout: u64,
    /// Does not download seeds (tests, no network); the copies in `<data-dir>/seeds` are not used.
    #[arg(long)]
    no_seed: bool,
    /// Distinct clients from distinct IPs needed to publish a record (1 publishes at once).
    #[arg(long, default_value_t = 2, value_parser = clap::value_parser!(u32).range(1..=32))]
    quorum: u32,
    /// Seconds between saves of the state and exports.
    #[arg(long, default_value_t = 300)]
    save_interval: u64,
    /// Header with the client IP set by the reverse proxy (e.g. X-Real-IP). When set it is
    /// required (400 without it) and replaces the TCP peer address for quorum and limits.
    #[arg(long)]
    real_ip_header: Option<String>,
    /// Requests per IP in each window.
    #[arg(long, default_value_t = 30)]
    ip_requests: u32,
    /// Window of the per-IP request limit, in seconds.
    #[arg(long, default_value_t = 600)]
    ip_window: u64,
    /// New records per client and day.
    #[arg(long, default_value_t = 20_000)]
    client_daily: u64,
    /// New records per IP and day.
    #[arg(long, default_value_t = 100_000)]
    ip_daily: u64,
    /// Records (published and pending) per partition.
    #[arg(long, default_value_t = 200_000)]
    partition_cap: usize,
    /// Partitions (title and version) the server accepts.
    #[arg(long, default_value_t = 500)]
    max_partitions: usize,
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        if let Ok(mut term) = signal(SignalKind::terminate()) {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
            return;
        }
    }
    let _ = tokio::signal::ctrl_c().await;
}

async fn save(app: &Arc<App>, dir: &Arc<PathBuf>) {
    let (app, dir) = (app.clone(), dir.clone());
    let result = tokio::task::spawn_blocking(move || {
        let mut store = app.store();
        if !store.is_dirty() {
            return Ok(false);
        }
        store.save(&dir).map(|()| true)
    })
    .await;
    match result {
        Ok(Ok(true)) => tracing::info!("state saved"),
        Ok(Ok(false)) => {}
        Ok(Err(e)) => tracing::error!("saving failed: {e}"),
        Err(e) => tracing::error!("saving task failed: {e}"),
    }
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stdout()))
        .init();
    let args = Args::parse();
    let settings = Settings {
        quorum: args.quorum as usize,
        partition_cap: args.partition_cap,
        max_partitions: args.max_partitions,
        default_release: args.default_release.clone(),
        ..Settings::default()
    };
    let mut store = match Store::load(settings, &args.data_dir) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("cannot load the state from {}: {e}", args.data_dir.display());
            return std::process::ExitCode::FAILURE;
        }
    };
    let started = std::time::Instant::now();
    let mut releases: Vec<Release> = Vec::new();
    if let Some(dir) = &args.known {
        match store::load_known(dir) {
            Ok(known) => releases = known,
            Err(e) => {
                tracing::error!("cannot read the known manifests from {}: {e}", dir.display());
                return std::process::ExitCode::FAILURE;
            }
        }
    }
    if !args.no_seed {
        let components = [&args.seed_repo, &args.seed_ref, &args.seed_path];
        if !components.iter().all(|c| seed::valid_component(c)) {
            tracing::error!("invalid --seed-repo, --seed-ref or --seed-path");
            return std::process::ExitCode::FAILURE;
        }
        let tag = Some(args.default_release.clone())
            .filter(|t| t != store::LOOSE_RELEASE && seed::valid_component(t))
            .filter(|t| !releases.iter().any(|r| &r.name == t));
        let cfg = seed::Config {
            api: args.seed_api.clone(),
            repo: args.seed_repo.clone(),
            git_ref: args.seed_ref.clone(),
            path: args.seed_path.clone(),
            tag,
            dir: args.data_dir.join("seeds"),
            timeout: Duration::from_secs(args.seed_timeout),
        };
        let folders = tokio::task::spawn_blocking(move || seed::refresh(&seed::HttpFetcher, &cfg)).await.unwrap_or_default();
        for (name, folder) in folders {
            if releases.iter().any(|r| r.name == name) {
                tracing::warn!("release {name} is in --known: the downloaded seeds of that name are not used");
                continue;
            }
            match store::load_release_dir(&folder) {
                Ok(parts) => releases.push(Release { name, parts }),
                Err(e) => tracing::warn!("cannot read the seeds in {}: {e}", folder.display()),
            }
        }
    }
    for release in &releases {
        let records: usize = release.parts.values().map(|k| k.len()).sum();
        tracing::info!("release {}: {} partitions, {records} records", release.name, release.parts.len());
    }
    // Pipelines that no draw creates from the modules they name (records of recordings that
    // mixed up their modules) are neither published nor served, whatever brought them.
    for (key, unlinked) in store.set_known(releases) {
        tracing::warn!(
            "seeds {}_{}: left out {} pipelines that no draw creates from their modules, such as {}",
            key.title,
            key.version,
            unlinked.count,
            unlinked.example.as_deref().unwrap_or("-")
        );
    }
    let removed = store.drop_unlinkable();
    for (key, published, pending, unlinked) in &removed {
        tracing::warn!(
            "state {}_{}: removed {published} published and {pending} pending pipelines that no draw creates from \
             their modules, such as {}",
            key.title,
            key.version,
            unlinked.example.as_deref().unwrap_or("-")
        );
    }
    if !removed.is_empty() {
        // The state is written again without them at once; the one before is kept next to it.
        let state = args.data_dir.join("state.json");
        let backup = args.data_dir.join(format!("state.json.before-unlinked-{}", http::unix_now()));
        if state.exists() {
            if let Err(e) = std::fs::copy(&state, &backup) {
                tracing::error!("cannot back up {} to {}: {e}", state.display(), backup.display());
                return std::process::ExitCode::FAILURE;
            }
        }
        if let Err(e) = store.save(&args.data_dir) {
            tracing::error!("saving the cleaned state failed: {e}");
            return std::process::ExitCode::FAILURE;
        }
        let total: usize = removed.iter().map(|(_, published, pending, _)| published + pending).sum();
        tracing::info!("state cleaned: {total} pipelines removed, previous state in {}", backup.display());
    }
    if !store.has_release(&args.default_release) && args.default_release != store::LOOSE_RELEASE {
        tracing::warn!(
            "the default release {} is not loaded (releases: {:?}): old clients get everything",
            args.default_release,
            store.release_names()
        );
    }
    tracing::info!("seeds ready in {:.1} s", started.elapsed().as_secs_f64());
    let limits = Limits::new(LimitConfig {
        ip_requests: args.ip_requests,
        ip_window_secs: args.ip_window,
        client_daily: args.client_daily,
        ip_daily: args.ip_daily,
    });
    let app = Arc::new(App::new(store, limits, args.real_ip_header.clone()));
    let dir = Arc::new(args.data_dir.clone());

    let listener = match tokio::net::TcpListener::bind(args.listen).await {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("cannot listen on {}: {e}", args.listen);
            return std::process::ExitCode::FAILURE;
        }
    };
    tracing::info!(
        "listening on {} (quorum {}, real ip header {})",
        args.listen,
        args.quorum,
        args.real_ip_header.as_deref().unwrap_or("none: TCP peer")
    );

    let saver = {
        let (app, dir) = (app.clone(), dir.clone());
        let every = Duration::from_secs(args.save_interval.max(1));
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(every);
            tick.tick().await;
            loop {
                tick.tick().await;
                save(&app, &dir).await;
                let now = http::unix_now();
                app.limits_gc(now);
            }
        })
    };

    let served = axum::serve(
        listener,
        http::router(app.clone()).into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await;
    saver.abort();
    if let Err(e) = served {
        tracing::error!("server error: {e}");
    }
    tracing::info!("shutting down");
    save(&app, &dir).await;
    std::process::ExitCode::SUCCESS
}
