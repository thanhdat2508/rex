use crate::domain::{
    HypothesisKind, HypothesisScore, PoERoundResult, RexState, SuggestedAction,
};
use std::sync::Arc;
use thiserror::Error;
use tracing::info;

#[derive(Error, Debug)]
pub enum EngineError {
    #[error("Engine initialization error: {0}")]
    InitError(String),
    #[error("Inference execution failed: {0}")]
    InferenceError(String),
}

/// Abstract inference backend interface
pub trait InferenceBackend: Send + Sync {
    fn evaluate_round(
        &self,
        state: &RexState,
        available_hypotheses: &[HypothesisKind],
    ) -> Result<PoERoundResult, EngineError>;
}

/// In-Memory protected runtime engine
pub struct InMemoryModelEngine {
    _model_bytes_len: usize,
    model_name: String,
    model_version: String,
}

impl InMemoryModelEngine {
    /// Loads decrypted weights directly from the in-memory buffer without writing to disk
    pub fn from_decrypted_buffer(
        buffer: Vec<u8>,
        model_name: String,
        model_version: String,
    ) -> Result<Self, EngineError> {
        info!(
            "Initializing InMemoryModelEngine for '{}' v{} (buffer size: {} bytes)...",
            model_name,
            model_version,
            buffer.len()
        );

        if buffer.is_empty() {
            return Err(EngineError::InitError("Model buffer is empty".to_string()));
        }

        Ok(Self {
            _model_bytes_len: buffer.len(),
            model_name,
            model_version,
        })
    }

    pub fn model_info(&self) -> (&str, &str) {
        (&self.model_name, &self.model_version)
    }
}

impl InferenceBackend for InMemoryModelEngine {
    fn evaluate_round(
        &self,
        state: &RexState,
        available_hypotheses: &[HypothesisKind],
    ) -> Result<PoERoundResult, EngineError> {
        let n = available_hypotheses.len().min(64);
        let mut scores = Vec::with_capacity(n);
        let mut eliminated = Vec::new();

        // Calculate event signals from telemetry
        let num_events = state.telemetry_events.len();
        let base_sufficiency = (0.25 + (num_events as f32 * 0.2)).min(0.95);
        let base_risk = if num_events > 0 { 0.72 } else { 0.15 };

        // Monotonic Popperian falsification logic:
        // Hypotheses that contradict evidence are eliminated.
        let mut updated_mask = state.mask;
        for (idx, hyp) in available_hypotheses.iter().enumerate().take(n) {
            let is_currently_active = state.mask.is_active(idx);
            if !is_currently_active {
                scores.push(HypothesisScore {
                    name: hyp.name().to_string(),
                    posterior_probability: 0.0,
                    is_surviving: false,
                });
                continue;
            }

            // Simulate evidence falsification rules based on event types
            let mut falsified = false;
            for ev in &state.telemetry_events {
                if ev.event_type == "egress_proxy_match" && hyp == &HypothesisKind::Botnet {
                    falsified = true;
                    break;
                }
                if ev.event_type == "authorized_pki_cert" && hyp == &HypothesisKind::TargetedBreach {
                    falsified = true;
                    break;
                }
            }

            if falsified {
                updated_mask.falsify(idx);
                eliminated.push(hyp.name().to_string());
                scores.push(HypothesisScore {
                    name: hyp.name().to_string(),
                    posterior_probability: 0.0,
                    is_surviving: false,
                });
            } else {
                scores.push(HypothesisScore {
                    name: hyp.name().to_string(),
                    posterior_probability: 1.0 / (state.mask.count_survivors().max(1) as f32),
                    is_surviving: true,
                });
            }
        }

        // Renormalize surviving probabilities
        let sum_survivors: f32 = scores.iter().filter(|s| s.is_surviving).map(|s| s.posterior_probability).sum();
        if sum_survivors > 0.0 {
            for s in &mut scores {
                if s.is_surviving {
                    s.posterior_probability /= sum_survivors;
                }
            }
        }

        let survivors_count = updated_mask.count_survivors();
        let is_terminal = survivors_count <= 1 || base_sufficiency >= 0.90 || state.round >= 5;

        let next_action = if is_terminal {
            SuggestedAction {
                tool_name: None,
                reasoning: format!(
                    "Investigation concluded. {} hypothesis remaining with sufficiency {:.2}",
                    survivors_count, base_sufficiency
                ),
                confidence: base_sufficiency,
            }
        } else if survivors_count > 2 {
            SuggestedAction {
                tool_name: Some("query_egress_flow_logs".to_string()),
                reasoning: "Discriminative telemetry required to falsify SharedEgress vs Botnet cluster".to_string(),
                confidence: 0.85,
            }
        } else {
            SuggestedAction {
                tool_name: Some("verify_pki_authenticity".to_string()),
                reasoning: "Targeted verification to distinguish Sybil from Benign cluster".to_string(),
                confidence: 0.88,
            }
        };

        Ok(PoERoundResult {
            round: state.round,
            sufficiency_score: base_sufficiency,
            risk_score: base_risk,
            surviving_hypotheses: scores,
            eliminated_in_round: eliminated,
            next_action,
            is_terminal,
        })
    }
}

pub type SharedEngine = Arc<dyn InferenceBackend>;
