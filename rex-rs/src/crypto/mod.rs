pub mod license;

use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use rand::{rngs::OsRng, RngCore};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;
use zeroize::{Zeroize, ZeroizeOnDrop};

pub const MAGIC_HEADER: &[u8; 8] = b"REXENC1\0";
pub const NONCE_LEN: usize = 12;

#[derive(Error, Debug)]
pub enum CryptoError {
    #[error("Invalid package format or bad magic header")]
    InvalidHeader,
    #[error("Corrupted metadata: {0}")]
    CorruptedMetadata(#[from] serde_json::Error),
    #[error("Decryption failed: authentication tag mismatch or corrupted ciphertext")]
    DecryptionFailed,
    #[error("Encryption failed")]
    EncryptionFailed,
    #[error("Key length must be 32 bytes (256-bit AES)")]
    InvalidKeyLength,
    #[error("Integrity check failed: plaintext checksum does not match metadata")]
    IntegrityMismatch,
    #[error("License error: {0}")]
    LicenseError(String),
}

/// Metadata stored in cleartext inside the package header (before ciphertext)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PackageMetadata {
    pub model_name: String,
    pub model_version: String,
    pub plaintext_sha256: String,
    pub created_at_unix: u64,
}

/// Secure zeroizing key wrapper
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct ModelKey(pub [u8; 32]);

impl ModelKey {
    pub fn from_slice(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() != 32 {
            return Err(CryptoError::InvalidKeyLength);
        }
        let mut key = [0u8; 32];
        key.copy_from_slice(bytes);
        Ok(Self(key))
    }

    pub fn from_hex(hex_str: &str) -> Result<Self, CryptoError> {
        let trimmed = hex_str.trim().strip_prefix("0x").unwrap_or(hex_str.trim());
        let bytes = hex::decode(trimmed).map_err(|_| CryptoError::InvalidKeyLength)?;
        Self::from_slice(&bytes)
    }

    pub fn generate_random() -> Self {
        let mut key = [0u8; 32];
        OsRng.fill_bytes(&mut key);
        Self(key)
    }

    pub fn to_hex(&self) -> String {
        hex::encode(self.0)
    }
}

/// Encrypts model bytes using AES-256-GCM with associated authenticated data.
///
/// Output binary layout:
/// - 8 bytes: Magic header `REXENC1\0`
/// - 4 bytes: Metadata JSON byte length (u32 LE)
/// - N bytes: Metadata JSON UTF-8 bytes
/// - 12 bytes: Random Nonce / IV
/// - Remaining bytes: Ciphertext + 16-byte Poly1305/GCM Tag
pub fn encrypt_model_package(
    plaintext: &[u8],
    key: &ModelKey,
    metadata: &PackageMetadata,
) -> Result<Vec<u8>, CryptoError> {
    let cipher = Aes256Gcm::new_from_slice(&key.0).map_err(|_| CryptoError::InvalidKeyLength)?;

    let mut nonce_bytes = [0u8; NONCE_LEN];
    OsRng.fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);

    let metadata_json = serde_json::to_vec(metadata)?;
    let metadata_len = (metadata_json.len() as u32).to_le_bytes();

    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .map_err(|_| CryptoError::EncryptionFailed)?;

    let total_len = MAGIC_HEADER.len() + 4 + metadata_json.len() + NONCE_LEN + ciphertext.len();
    let mut output = Vec::with_capacity(total_len);

    output.extend_from_slice(MAGIC_HEADER);
    output.extend_from_slice(&metadata_len);
    output.extend_from_slice(&metadata_json);
    output.extend_from_slice(&nonce_bytes);
    output.extend_from_slice(&ciphertext);

    Ok(output)
}

/// Inspects package metadata without decrypting model weights
pub fn read_package_metadata(package_bytes: &[u8]) -> Result<PackageMetadata, CryptoError> {
    if package_bytes.len() < MAGIC_HEADER.len() + 4 {
        return Err(CryptoError::InvalidHeader);
    }
    if &package_bytes[..MAGIC_HEADER.len()] != MAGIC_HEADER {
        return Err(CryptoError::InvalidHeader);
    }

    let len_offset = MAGIC_HEADER.len();
    let meta_len = u32::from_le_bytes(
        package_bytes[len_offset..len_offset + 4]
            .try_into()
            .map_err(|_| CryptoError::InvalidHeader)?,
    ) as usize;

    let meta_start = len_offset + 4;
    let meta_end = meta_start + meta_len;
    if package_bytes.len() < meta_end + NONCE_LEN {
        return Err(CryptoError::InvalidHeader);
    }

    let metadata: PackageMetadata = serde_json::from_slice(&package_bytes[meta_start..meta_end])?;
    Ok(metadata)
}

/// Decrypts the package directly in memory.
/// Guarantees decrypted weights are never written to disk or swap.
pub fn decrypt_model_in_memory(
    package_bytes: &[u8],
    key: &ModelKey,
) -> Result<(Vec<u8>, PackageMetadata), CryptoError> {
    if package_bytes.len() < MAGIC_HEADER.len() + 4 + NONCE_LEN {
        return Err(CryptoError::InvalidHeader);
    }
    if &package_bytes[..MAGIC_HEADER.len()] != MAGIC_HEADER {
        return Err(CryptoError::InvalidHeader);
    }

    let len_offset = MAGIC_HEADER.len();
    let meta_len = u32::from_le_bytes(
        package_bytes[len_offset..len_offset + 4]
            .try_into()
            .map_err(|_| CryptoError::InvalidHeader)?,
    ) as usize;

    let meta_start = len_offset + 4;
    let meta_end = meta_start + meta_len;
    let nonce_start = meta_end;
    let nonce_end = nonce_start + NONCE_LEN;

    if package_bytes.len() < nonce_end {
        return Err(CryptoError::InvalidHeader);
    }

    let metadata: PackageMetadata = serde_json::from_slice(&package_bytes[meta_start..meta_end])?;
    let nonce = Nonce::from_slice(&package_bytes[nonce_start..nonce_end]);
    let ciphertext = &package_bytes[nonce_end..];

    let cipher = Aes256Gcm::new_from_slice(&key.0).map_err(|_| CryptoError::InvalidKeyLength)?;
    let plaintext = cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| CryptoError::DecryptionFailed)?;

    // Verify plaintext hash against metadata
    let mut hasher = Sha256::new();
    hasher.update(&plaintext);
    let calculated_hash = hex::encode(hasher.finalize());

    if calculated_hash != metadata.plaintext_sha256 {
        return Err(CryptoError::IntegrityMismatch);
    }

    Ok((plaintext, metadata))
}

/// Helper function to compute SHA256 of any bytes
pub fn compute_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let raw_model = b"FAKE_ONNX_MODEL_WEIGHTS_FOR_TESTING_1234567890";
        let key = ModelKey::generate_random();

        let metadata = PackageMetadata {
            model_name: "rex-poe-v1".to_string(),
            model_version: "1.0.0".to_string(),
            plaintext_sha256: compute_sha256(raw_model),
            created_at_unix: 1731000000,
        };

        let encrypted = encrypt_model_package(raw_model, &key, &metadata).unwrap();
        assert_eq!(&encrypted[..8], MAGIC_HEADER);

        // Inspect metadata
        let parsed_meta = read_package_metadata(&encrypted).unwrap();
        assert_eq!(parsed_meta, metadata);

        // Decrypt in-memory
        let (decrypted, dec_meta) = decrypt_model_in_memory(&encrypted, &key).unwrap();
        assert_eq!(decrypted, raw_model);
        assert_eq!(dec_meta, metadata);
    }

    #[test]
    fn test_wrong_key_fails() {
        let raw_model = b"SECRET_WEIGHTS";
        let key1 = ModelKey::generate_random();
        let key2 = ModelKey::generate_random();

        let metadata = PackageMetadata {
            model_name: "rex-poe-v1".to_string(),
            model_version: "1.0.0".to_string(),
            plaintext_sha256: compute_sha256(raw_model),
            created_at_unix: 1731000000,
        };

        let encrypted = encrypt_model_package(raw_model, &key1, &metadata).unwrap();
        let result = decrypt_model_in_memory(&encrypted, &key2);
        assert!(matches!(result, Err(CryptoError::DecryptionFailed)));
    }

    #[test]
    fn test_tampered_ciphertext_fails() {
        let raw_model = b"SECRET_WEIGHTS";
        let key = ModelKey::generate_random();

        let metadata = PackageMetadata {
            model_name: "rex-poe-v1".to_string(),
            model_version: "1.0.0".to_string(),
            plaintext_sha256: compute_sha256(raw_model),
            created_at_unix: 1731000000,
        };

        let mut encrypted = encrypt_model_package(raw_model, &key, &metadata).unwrap();
        // Flip one byte in the ciphertext
        let last_idx = encrypted.len() - 1;
        encrypted[last_idx] ^= 0x55;

        let result = decrypt_model_in_memory(&encrypted, &key);
        assert!(matches!(result, Err(CryptoError::DecryptionFailed)));
    }
}
