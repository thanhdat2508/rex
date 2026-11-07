use clap::Parser;
use rex_rs::api::{create_router, AppState};
use rex_rs::crypto::{
    decrypt_model_in_memory, license::LicenseManager, read_package_metadata,
};
use rex_rs::engine::InMemoryModelEngine;
use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Parser, Debug)]
#[command(
    name = "rex-runtime",
    about = "Secure high-performance on-premise runtime for Rex AI models"
)]
struct Args {
    /// Path to encrypted model package (.rexenc)
    #[arg(short, long, env = "REX_MODEL_PATH")]
    model_path: PathBuf,

    /// Direct 256-bit AES decryption key in hex (optional if using license server)
    #[arg(short, long, env = "REX_MODEL_KEY")]
    key: Option<String>,

    /// Customer license key string for server handshake
    #[arg(short = 'l', long, env = "REX_LICENSE_KEY")]
    license_key: Option<String>,

    /// Remote license server URL
    #[arg(short = 's', long, env = "REX_LICENSE_SERVER")]
    license_server: Option<String>,

    /// Optional API key / Bearer token required for clients
    #[arg(long, env = "REX_API_KEY")]
    api_key: Option<String>,

    /// Bind IP address
    #[arg(long, default_value = "0.0.0.0", env = "REX_HOST")]
    host: String,

    /// Bind TCP port
    #[arg(short, long, default_value_t = 8080, env = "REX_PORT")]
    port: u16,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "rex_rs=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    info!("Starting Rex On-Premise Secure Runtime...");
    info!("Loading encrypted package from: {}", args.model_path.display());

    let package_bytes = fs::read(&args.model_path).map_err(|e| {
        error!("Failed to read model package file: {}", e);
        e
    })?;

    // Validate header and print metadata
    let metadata = read_package_metadata(&package_bytes)?;
    info!(
        "Package verified: '{}' (version {}), checksum {}",
        metadata.model_name, metadata.model_version, metadata.plaintext_sha256
    );

    // Resolve decryption key (direct or license handshake)
    let license_mgr = Arc::new(LicenseManager::new(args.license_server, args.api_key));
    let model_key = license_mgr
        .resolve_model_key(args.key.as_deref(), args.license_key.as_deref())
        .await?;

    // Decrypt directly into memory (zero-disk)
    info!("Decrypting model package in memory (zero disk footprint)...");
    let (decrypted_weights, meta) = decrypt_model_in_memory(&package_bytes, &model_key)?;
    info!(
        "Decryption successful. Decrypted weights: {} bytes. Authenticated tag verified.",
        decrypted_weights.len()
    );

    // Initialize in-memory runtime engine
    let engine = InMemoryModelEngine::from_decrypted_buffer(
        decrypted_weights,
        meta.model_name.clone(),
        meta.model_version.clone(),
    )?;

    let startup_time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs();

    let state = AppState {
        engine: Arc::new(engine),
        license_mgr,
        model_name: meta.model_name,
        model_version: meta.model_version,
        startup_time,
    };

    let router = create_router(state);
    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;

    info!("Rex Runtime HTTP & SSE service listening on http://{}", addr);
    info!("  - Health check:     GET  /health");
    info!("  - Investigation:    POST /api/v1/investigate");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await?;

    Ok(())
}
