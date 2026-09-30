mod api;
mod config;
mod metrics;
mod providers;
mod router;

use config::Config;
use router::classifier::JevClassifier;
use router::RouterService;
use std::path::PathBuf;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info,route=info".into()))
        .init();

    let path: PathBuf = std::env::args().nth(1).or_else(|| std::env::var("ROUTER_CONFIG").ok()).unwrap_or_else(|| "router.toml".into()).into();
    let cfg = Config::load(&path).unwrap_or_else(|e| {
        eprintln!("config error: {e}");
        std::process::exit(2);
    });

    let jev = JevClassifier::new(cfg.jev.clone());
    if !jev.has_key() {
        tracing::warn!("{} is not set: every request will be routed to STANDARD", cfg.jev.api_key_env);
    }
    let svc = RouterService::build(&cfg, Arc::new(jev)).unwrap_or_else(|e| {
        eprintln!("startup error: {e}");
        std::process::exit(2);
    });
    for (id, h) in svc.provider_health().await {
        tracing::info!(provider = %id, health = ?h, "provider");
    }

    let listener = tokio::net::TcpListener::bind(&cfg.server.listen).await.unwrap_or_else(|e| {
        eprintln!("cannot bind {}: {e}", cfg.server.listen);
        std::process::exit(2);
    });
    tracing::info!("listening on http://{}/v1", cfg.server.listen);
    axum::serve(listener, api::app(svc)).with_graceful_shutdown(async { tokio::signal::ctrl_c().await.ok(); }).await.unwrap();
}
