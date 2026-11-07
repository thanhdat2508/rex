use axum::{
    body::Body,
    http::{header, Request, StatusCode},
};
use http_body_util::BodyExt;
use rex_rs::api::{create_router, AppState, HealthResponse, InvestigateResponse};
use rex_rs::crypto::{
    compute_sha256, decrypt_model_in_memory, encrypt_model_package, ModelKey, PackageMetadata,
};

use rex_rs::engine::InMemoryModelEngine;
use std::sync::Arc;
use tower::ServiceExt;

#[tokio::test]
async fn test_full_encrypted_runtime_lifecycle() {
    // 1. Pack / Encrypt mock model artifact
    let raw_model = b"ACTUAL_ONNX_MODEL_TENSOR_BYTES_STREAM_42";
    let key = ModelKey::generate_random();

    let metadata = PackageMetadata {
        model_name: "rex-poe-investigator".to_string(),
        model_version: "2.1.0".to_string(),
        plaintext_sha256: compute_sha256(raw_model),
        created_at_unix: 1731000000,
    };

    let encrypted_pkg = encrypt_model_package(raw_model, &key, &metadata).unwrap();

    // 2. Decrypt in memory (zero-disk)
    let (decrypted_bytes, meta) = decrypt_model_in_memory(&encrypted_pkg, &key).unwrap();
    assert_eq!(decrypted_bytes, raw_model);

    // 3. Initialize Engine
    let engine = InMemoryModelEngine::from_decrypted_buffer(
        decrypted_bytes,
        meta.model_name.clone(),
        meta.model_version.clone(),
    )
    .unwrap();

    let state = AppState {
        engine: Arc::new(engine),
        model_name: meta.model_name,
        model_version: meta.model_version,
        startup_time: 1731000000,
    };

    let app = create_router(state);

    // 4. Test GET /health
    let req = Request::builder()
        .uri("/health")
        .method("GET")
        .body(Body::empty())
        .unwrap();

    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let health: HealthResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(health.status, "ok");
    assert!(health.model_loaded);
    assert_eq!(health.model_name, "rex-poe-investigator");

    // 5. Test POST /api/v1/investigate (JSON)
    let inv_payload = serde_json::json!({
        "target_entity": "198.51.100.24",
        "initial_events": [
            {
                "timestamp_unix": 1731000100,
                "event_type": "high_volume_syn",
                "summary": "Burst of SYN traffic towards port 443",
                "source": "ids-sensor-01"
            }
        ],
        "max_rounds": 3,
        "stream": false
    });

    let req = Request::builder()
        .uri("/api/v1/investigate")
        .method("POST")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&inv_payload).unwrap()))
        .unwrap();

    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = resp.into_body().collect().await.unwrap().to_bytes();
    let inv_resp: InvestigateResponse = serde_json::from_slice(&body).unwrap();
    assert_eq!(inv_resp.target_entity, "198.51.100.24");
    assert!(inv_resp.total_rounds >= 1);
    assert!(inv_resp.final_result.sufficiency_score > 0.0);

    // 6. Test POST /api/v1/investigate with SSE (Accept: text/event-stream)
    let sse_req = Request::builder()
        .uri("/api/v1/investigate")
        .method("POST")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::ACCEPT, "text/event-stream")
        .body(Body::from(serde_json::to_vec(&inv_payload).unwrap()))
        .unwrap();

    let sse_resp = app.oneshot(sse_req).await.unwrap();
    assert_eq!(sse_resp.status(), StatusCode::OK);
    assert!(sse_resp
        .headers()
        .get(header::CONTENT_TYPE)
        .unwrap()
        .to_str()
        .unwrap()
        .contains("text/event-stream"));
}
