use super::{CryptoError, ModelKey};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeRequest {
    pub license_key: String,
    pub machine_fingerprint: String,
    pub nonce: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChallengeResponse {
    pub session_token: String,
    pub encrypted_key_hex: String,
    pub expires_at: u64,
}

pub struct LicenseManager {
    client: Client,
    license_server_url: Option<String>,
}

impl LicenseManager {
    pub fn new(license_server_url: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_default();

        Self {
            client,
            license_server_url,
        }
    }

    /// Acquires the ModelKey either directly from an environment/config key
    /// or by performing a challenge-handshake with the license server.
    pub async fn resolve_model_key(
        &self,
        direct_key_hex: Option<&str>,
        license_key: Option<&str>,
    ) -> Result<ModelKey, CryptoError> {
        // 1. Direct key provided in config/env
        if let Some(key_str) = direct_key_hex {
            if !key_str.trim().is_empty() {
                info!("Acquired model decryption key from direct configuration.");
                return ModelKey::from_hex(key_str);
            }
        }

        // 2. License key via license server handshake
        if let (Some(server_url), Some(lic)) = (&self.license_server_url, license_key) {
            info!("Contacting license server {} for session decryption key...", server_url);
            return self.fetch_session_key(server_url, lic).await;
        }

        Err(CryptoError::LicenseError(
            "No valid decryption key provided. Set REX_MODEL_KEY or REX_LICENSE_KEY with a valid license server.".to_string(),
        ))
    }

    async fn fetch_session_key(
        &self,
        server_url: &str,
        license_key: &str,
    ) -> Result<ModelKey, CryptoError> {
        let req = ChallengeRequest {
            license_key: license_key.to_string(),
            machine_fingerprint: get_system_fingerprint(),
            nonce: hex::encode(rand::random::<[u8; 16]>()),
        };

        let endpoint = format!("{}/api/v1/license/challenge", server_url.trim_end_matches('/'));
        let res = self
            .client
            .post(&endpoint)
            .json(&req)
            .send()
            .await
            .map_err(|e| CryptoError::LicenseError(format!("Network request failed: {}", e)))?;

        if !res.status().is_success() {
            let status = res.status();
            let text = res.text().await.unwrap_or_default();
            return Err(CryptoError::LicenseError(format!(
                "License server rejected request (status {}): {}",
                status, text
            )));
        }

        let body: ChallengeResponse = res.json().await.map_err(|e| {
            CryptoError::LicenseError(format!("Invalid response from license server: {}", e))
        })?;

        ModelKey::from_hex(&body.encrypted_key_hex)
    }
}

/// Simple platform-neutral machine fingerprint
fn get_system_fingerprint() -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(std::env::consts::OS.as_bytes());
    hasher.update(std::env::consts::ARCH.as_bytes());
    if let Ok(hostname) = std::env::var("HOSTNAME") {
        hasher.update(hostname.as_bytes());
    }
    hex::encode(&hasher.finalize()[..16])
}
