use serde::{Deserialize, Serialize};

/// Fixed-size or extensible hypothesis in the Popperian falsification taxonomy
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HypothesisKind {
    Botnet,
    SybilCluster,
    SharedEgress,
    TargetedBreach,
    InsiderThreat,
    BenignAnomaly,
    Custom(String),
}

impl HypothesisKind {
    pub fn name(&self) -> &str {
        match self {
            Self::Botnet => "Botnet",
            Self::SybilCluster => "SybilCluster",
            Self::SharedEgress => "SharedEgress",
            Self::TargetedBreach => "TargetedBreach",
            Self::InsiderThreat => "InsiderThreat",
            Self::BenignAnomaly => "BenignAnomaly",
            Self::Custom(s) => s.as_str(),
        }
    }
}

/// Bitmask enforcing monotonic falsification:
/// Once a hypothesis bit is set to 0 (falsified), it cannot transition back to 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PoEMask(pub u64);

impl PoEMask {
    pub fn all_active(num_hypotheses: usize) -> Self {
        assert!(num_hypotheses <= 64, "Max 64 hypotheses supported in bitmask");
        let mask = if num_hypotheses == 64 {
            u64::MAX
        } else {
            (1u64 << num_hypotheses) - 1
        };
        Self(mask)
    }

    pub fn is_active(&self, idx: usize) -> bool {
        assert!(idx < 64);
        (self.0 & (1u64 << idx)) != 0
    }

    /// Monotonic falsification update:
    /// new_mask = current_mask AND incoming_mask
    pub fn falsify(&mut self, idx: usize) {
        assert!(idx < 64);
        self.0 &= !(1u64 << idx);
    }

    pub fn apply_mask(&mut self, other: PoEMask) {
        self.0 &= other.0;
    }

    pub fn count_survivors(&self) -> u32 {
        self.0.count_ones()
    }
}

/// The state of an ongoing investigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RexState {
    pub investigation_id: String,
    pub target_entity: String,
    pub telemetry_events: Vec<TelemetryEvent>,
    pub round: usize,
    pub mask: PoEMask,
    pub is_concluded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    pub timestamp_unix: u64,
    pub event_type: String,
    pub summary: String,
    pub source: String,
}

/// Result of evaluating a single Popperian elimination round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoERoundResult {
    pub round: usize,
    pub sufficiency_score: f32, // S in [0.0, 1.0]
    pub risk_score: f32,        // R in [0.0, 1.0]
    pub surviving_hypotheses: Vec<HypothesisScore>,
    pub eliminated_in_round: Vec<String>,
    pub next_action: SuggestedAction,
    pub is_terminal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisScore {
    pub name: String,
    pub posterior_probability: f32,
    pub is_surviving: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedAction {
    pub tool_name: Option<String>,
    pub reasoning: String,
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monotonic_falsification() {
        let mut mask = PoEMask::all_active(4);
        assert_eq!(mask.count_survivors(), 4);

        mask.falsify(1);
        assert_eq!(mask.count_survivors(), 3);
        assert!(!mask.is_active(1));
        assert!(mask.is_active(0));
        assert!(mask.is_active(2));
        assert!(mask.is_active(3));

        // Attempting to revive via OR is impossible with monotonic apply_mask
        let revival = PoEMask::all_active(4);
        mask.apply_mask(revival);
        // Bit 1 remains 0!
        assert_eq!(mask.count_survivors(), 3);
        assert!(!mask.is_active(1));
    }
}
