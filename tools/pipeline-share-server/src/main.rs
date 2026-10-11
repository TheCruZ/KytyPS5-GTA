use clap::Parser;
use pipeline_share_server::http::{self, App};
use pipeline_share_server::limits::{LimitConfig, Limits};
use pipeline_share_server::store::{self, Settings, Store};
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
    /// Directory with the `.kpm` files of the repository (`assets/pipelines`): already known.
    #[arg(long)]
    known: Option<PathBuf>,
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
        ..Settings::default()
    };
    let mut store = match Store::load(settings, &args.data_dir) {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("cannot load the state from {}: {e}", args.data_dir.display());
            return std::process::ExitCode::FAILURE;
        }
    };
    if let Some(dir) = &args.known {
        match store::load_known(dir) {
            Ok(known) => {
                let records: usize = known.values().map(|k| k.len()).sum();
                tracing::info!("known: {} partitions, {records} records from {}", known.len(), dir.display());
                store.set_known(known);
            }
            Err(e) => {
                tracing::error!("cannot read the known manifests from {}: {e}", dir.display());
                return std::process::ExitCode::FAILURE;
            }
        }
    }
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
