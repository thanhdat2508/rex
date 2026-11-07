use clap::{Parser, Subcommand};
use rex_rs::crypto::{
    compute_sha256, encrypt_model_package, read_package_metadata, ModelKey, PackageMetadata,
};
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

#[derive(Parser, Debug)]
#[command(
    name = "rex-pack",
    about = "Model encryption, packaging, and inspection CLI for Rex"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Encrypt an ONNX model file into a secure .rexenc package
    Encrypt {
        /// Path to raw ONNX model file
        #[arg(short, long)]
        input: PathBuf,

        /// Path to output encrypted package (.rexenc)
        #[arg(short, long)]
        output: PathBuf,

        /// Optional 256-bit AES key in hex (generated randomly if not provided)
        #[arg(short, long)]
        key: Option<String>,

        /// Model identifier name
        #[arg(long, default_value = "rex-model")]
        name: String,

        /// Model version string
        #[arg(long, default_value = "1.0.0")]
        version: String,
    },

    /// Inspect metadata of an encrypted package without decrypting
    Inspect {
        /// Path to encrypted package (.rexenc)
        #[arg(short, long)]
        input: PathBuf,
    },

    /// Generate a fresh random 256-bit AES key
    GenerateKey,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::GenerateKey => {
            let key = ModelKey::generate_random();
            println!("Generated AES-256 Key (Hex):");
            println!("{}", key.to_hex());
        }

        Commands::Inspect { input } => {
            let bytes = fs::read(&input)?;
            let meta = read_package_metadata(&bytes)?;
            println!("================ Encrypted Package Metadata ================");
            println!("File:             {}", input.display());
            println!("Package Size:     {} bytes", bytes.len());
            println!("Model Name:       {}", meta.model_name);
            println!("Model Version:    {}", meta.model_version);
            println!("Plaintext SHA256: {}", meta.plaintext_sha256);
            println!("Created Unix:     {}", meta.created_at_unix);
            println!("============================================================");
        }

        Commands::Encrypt {
            input,
            output,
            key,
            name,
            version,
        } => {
            let raw_bytes = fs::read(&input)?;
            let model_key = match key {
                Some(k) => ModelKey::from_hex(&k)?,
                None => {
                    let k = ModelKey::generate_random();
                    println!("No key provided. Generated new AES-256 Key (SAVE THIS):");
                    println!("==> {}", k.to_hex());
                    k
                }
            };

            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)?
                .as_secs();

            let metadata = PackageMetadata {
                model_name: name,
                model_version: version,
                plaintext_sha256: compute_sha256(&raw_bytes),
                created_at_unix: now,
            };

            let encrypted = encrypt_model_package(&raw_bytes, &model_key, &metadata)?;
            fs::write(&output, &encrypted)?;

            println!("Successfully encrypted model!");
            println!("Input:  {} ({} bytes)", input.display(), raw_bytes.len());
            println!("Output: {} ({} bytes)", output.display(), encrypted.len());
            println!("Plaintext SHA256: {}", metadata.plaintext_sha256);
        }
    }

    Ok(())
}
