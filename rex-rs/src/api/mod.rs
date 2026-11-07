use crate::crypto::license::SharedLicenseManager;
use crate::domain::{
    HypothesisKind, PoEMask, PoERoundResult, RexState, TelemetryEvent,
};
use crate::engine::SharedEngine;
use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::{convert::Infallible, time::Duration};
use tower_http::cors::CorsLayer;
use tracing::info;

#[derive(Clone)]
pub struct AppState {
    pub engine: SharedEngine,
    pub license_mgr: SharedLicenseManager,
    pub model_name: String,
    pub model_version: String,
    pub startup_time: u64,
}

#[derive(Debug, Deserialize)]
pub struct InvestigateRequest {
    pub investigation_id: Option<String>,
    pub target_entity: String,
    pub initial_events: Option<Vec<TelemetryEvent>>,
    pub max_rounds: Option<usize>,
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InvestigateResponse {
    pub investigation_id: String,
    pub target_entity: String,
    pub total_rounds: usize,
    pub final_result: PoERoundResult,
    pub history: Vec<PoERoundResult>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub model_loaded: bool,
    pub license_valid: bool,
    pub model_name: String,
    pub model_version: String,
    pub uptime_seconds: u64,
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/api/v1/investigate", post(investigate_handler))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

async fn health_handler(State(state): State<AppState>) -> Json<HealthResponse> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    Json(HealthResponse {
        status: "ok".to_string(),
        model_loaded: true,
        license_valid: state.license_mgr.is_valid(),
        model_name: state.model_name.clone(),
        model_version: state.model_version.clone(),
        uptime_seconds: now.saturating_sub(state.startup_time),
    })
}

async fn investigate_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<InvestigateRequest>,
) -> Response {
    // 1. Enforce license validity
    if let Err(e) = state.license_mgr.enforce_valid() {
        return (
            StatusCode::PAYMENT_REQUIRED,
            Json(serde_json::json!({
                "error": "License expired or invalid",
                "detail": e.to_string()
            })),
        )
            .into_response();
    }

    // 2. Verify authorization
    let auth_header = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok());
    if !state.license_mgr.verify_authorization(auth_header).await {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "Unauthorized: invalid Bearer token or API key" })),
        )
            .into_response();
    }

    let is_sse = payload.stream.unwrap_or(false)
        || headers
            .get(header::ACCEPT)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.contains("text/event-stream"))
            .unwrap_or(false);

    let investigation_id = payload
        .investigation_id
        .unwrap_or_else(|| format!("inv-{}", hex::encode(rand::random::<[u8; 8]>())));
    let target_entity = payload.target_entity;
    let initial_events = payload.initial_events.unwrap_or_default();
    let max_rounds = payload.max_rounds.unwrap_or(5);

    let hypotheses = vec![
        HypothesisKind::Botnet,
        HypothesisKind::SybilCluster,
        HypothesisKind::SharedEgress,
        HypothesisKind::TargetedBreach,
    ];

    if is_sse {
        info!("Streaming SSE investigation {} for {}", investigation_id, target_entity);
        let stream = run_investigation_stream(
            state.engine,
            investigation_id,
            target_entity,
            initial_events,
            hypotheses,
            max_rounds,
        );
        Sse::new(stream)
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
            .into_response()
    } else {
        info!("Executing synchronous investigation {} for {}", investigation_id, target_entity);
        match run_investigation_sync(
            &state.engine,
            investigation_id.clone(),
            target_entity.clone(),
            initial_events,
            &hypotheses,
            max_rounds,
        ) {
            Ok(resp) => Json(resp).into_response(),
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": e.to_string() })),
            )
                .into_response(),
        }
    }
}

fn run_investigation_sync(
    engine: &SharedEngine,
    investigation_id: String,
    target_entity: String,
    initial_events: Vec<TelemetryEvent>,
    hypotheses: &[HypothesisKind],
    max_rounds: usize,
) -> Result<InvestigateResponse, anyhow::Error> {
    let mut current_state = RexState {
        investigation_id: investigation_id.clone(),
        target_entity: target_entity.clone(),
        telemetry_events: initial_events,
        round: 1,
        mask: PoEMask::all_active(hypotheses.len()),
        is_concluded: false,
    };

    let mut history = Vec::new();
    let mut last_result = None;

    for r in 1..=max_rounds {
        current_state.round = r;
        let res = engine.evaluate_round(&current_state, hypotheses)?;
        let is_terminal = res.is_terminal;
        history.push(res.clone());
        last_result = Some(res);

        if is_terminal {
            current_state.is_concluded = true;
            break;
        }

        // Simulate acquiring telemetry suggested by Popperian engine
        current_state.telemetry_events.push(TelemetryEvent {
            timestamp_unix: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            event_type: "egress_proxy_match".to_string(),
            summary: "Corporate egress gateway forwarder verified".to_string(),
            source: "netflow-sensor".to_string(),
        });
    }

    let final_res = last_result.ok_or_else(|| anyhow::anyhow!("No rounds evaluated"))?;

    Ok(InvestigateResponse {
        investigation_id,
        target_entity,
        total_rounds: history.len(),
        final_result: final_res,
        history,
    })
}

fn run_investigation_stream(
    engine: SharedEngine,
    investigation_id: String,
    target_entity: String,
    initial_events: Vec<TelemetryEvent>,
    hypotheses: Vec<HypothesisKind>,
    max_rounds: usize,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let mut current_state = RexState {
            investigation_id: investigation_id.clone(),
            target_entity: target_entity.clone(),
            telemetry_events: initial_events,
            round: 1,
            mask: PoEMask::all_active(hypotheses.len()),
            is_concluded: false,
        };

        for r in 1..=max_rounds {
            current_state.round = r;
            match engine.evaluate_round(&current_state, &hypotheses) {
                Ok(round_res) => {
                    let is_terminal = round_res.is_terminal;
                    let json_str = serde_json::to_string(&round_res).unwrap_or_default();
                    yield Ok(Event::default().event("round").data(json_str));

                    if is_terminal {
                        yield Ok(Event::default().event("concluded").data("{\"status\":\"finished\"}"));
                        break;
                    }

                    // Append next evidence
                    tokio::time::sleep(Duration::from_millis(150)).await;
                    current_state.telemetry_events.push(TelemetryEvent {
                        timestamp_unix: std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs(),
                        event_type: "egress_proxy_match".to_string(),
                        summary: "Telemetry evidence matched proxy pool".to_string(),
                        source: "proxy-probe".to_string(),
                    });
                }
                Err(e) => {
                    yield Ok(Event::default().event("error").data(format!("{{\"error\":\"{}\"}}", e)));
                    break;
                }
            }
        }
    }
}
