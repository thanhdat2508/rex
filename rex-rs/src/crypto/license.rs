use super::{CryptoError, ModelKey};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
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
    expires_at: AtomicU64,
    session_token: RwLock<Option<String>>,
    required_api_key: Option<String>,
}

impl LicenseManager {
    pub fn new(license_server_url: Option<String>, required_api_key: Option<String>) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_default();

        Self {
            client,
            license_server_url,
            expires_at: AtomicU64::new(0),
            session_token: RwLock::new(None),
            required_api_key,
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
                // Static direct key has indefinite expiry (0)
                self.expires_at.store(0, Ordering::Relaxed);
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

        // Store active expiry and session token
        self.expires_at.store(body.expires_at, Ordering::Relaxed);
        let mut token_lock = self.session_token.write().await;
        *token_lock = Some(body.session_token);

        ModelKey::from_hex(&body.encrypted_key_hex)
    }

    /// Checks if the current license is within its validity period.
    pub fn is_valid(&self) -> bool {
        let exp = self.expires_at.load(Ordering::Relaxed);
        if exp == 0 {
            return true; // No expiration constraint configured
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now < exp
    }

    /// Enforces license validity; returns an error if license has expired.
    pub fn enforce_valid(&self) -> Result<(), CryptoError> {
        if !self.is_valid() {
            let exp = self.expires_at.load(Ordering::Relaxed);
            return Err(CryptoError::LicenseError(format!(
                "License expired at unix timestamp {}",
                exp
            )));
        }
        Ok(())
    }

    /// Verifies Authorization header (Bearer token or configured API key).
    pub async fn verify_authorization(&self, auth_header: Option<&str>) -> bool {
        let token_guard = self.session_token.read().await;
        let expected_token = self.required_api_key.as_deref().or(token_guard.as_deref());

        if let Some(expected) = expected_token {
            match auth_header {
                Some(header_val) => {
                    let provided = header_val
                        .strip_prefix("Bearer ")
                        .unwrap_or(header_val)
                        .trim();
                    provided == expected
                }
                None => false,
            }
        } else {
            true // No authentication required if neither API key nor session token is configured
        }
    }
}

pub type SharedLicenseManager = Arc<LicenseManager>;

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
