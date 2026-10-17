"""MECE (Mutually Exclusive, Collectively Exhaustive) Risk Hypothesis Framework.

Replaces hundreds of brittle granular event types with 4 foundational physical root causes:
  - H_botnet: Automated scripts, headless browsers, or residential proxy botnets.
  - H_sybil: Coordinated human operations with cloned profiles or virtual machines.
  - H_shared: Legitimate users on shared egress (mobile CGNAT, public Wi-Fi, office proxy).
  - H_novel: Uncataloged zero-day abuse, deepfake bypass, or novel business logic exploit.

All investigations are conditioned on a single task anchor token: <TASK_RISK_INVESTIGATION>.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Dict, List, Optional, Union

# Single unified task routing tokens
TASK_TOKEN_INVESTIGATION = "<TASK_RISK_INVESTIGATION>"
TASK_TOKEN_AUDIT = "<TS_AUDIT>"
DEFAULT_TASK_TOKEN = TASK_TOKEN_INVESTIGATION

ALL_TASK_TOKENS: List[str] = [TASK_TOKEN_INVESTIGATION, TASK_TOKEN_AUDIT]


class RiskHypothesis(str, Enum):
    """The 4 MECE physical root-cause hypotheses for digital abuse and attacks."""
    BOTNET = "H_botnet"
    SYBIL = "H_sybil"
    SHARED = "H_shared"
    NOVEL = "H_novel"


# Canonical hypothesis strings calibrated to 10-25 tokens for balanced self-attention
MECE_HYPOTHESIS_STRINGS: Dict[RiskHypothesis, str] = {
    RiskHypothesis.BOTNET: (
        "Automated execution via scripts, headless browsers, or residential botnets"
    ),
    RiskHypothesis.SYBIL: (
        "Coordinated human fraud operation using cloned profiles or virtual environments"
    ),
    RiskHypothesis.SHARED: (
        "Legitimate users traversing shared egress infrastructure such as mobile CGNAT or public WiFi"
    ),
    RiskHypothesis.NOVEL: (
        "Uncataloged zero-day abuse pattern, deepfake bypass, or novel business logic exploit"
    ),
}

# Detailed physical metadata for triage, mitigation, falsification rules, and explainability
HYPOTHESIS_METADATA: Dict[RiskHypothesis, Dict[str, Any]] = {
    RiskHypothesis.BOTNET: {
        "title": "Automation",
        "is_residual": False,
        "mechanism": "Script, curl, Puppeteer, Headless Chrome, botnet dân cư thực thi không có người dùng trực tiếp.",
        "signals": ["unusual_user_agent", "zero_mouse_movement", "burst_rate", "headless_fingerprint"],
        "falsification_criteria": [
            "mouse_curvature_entropy_high",
            "organic_keystroke_jitter_present",
            "residential_clean_ip",
        ],
        "allowed_tool_slots": ["SLOT_NETWORK", "SLOT_DEVICE", "SLOT_BEHAVIOR"],
        "primary_action": "block_ip_or_require_proof_of_work",
    },
    RiskHypothesis.SYBIL: {
        "title": "Human Farm",
        "is_residual": False,
        "mechanism": "Individuals operate in an organized manner using multiple virtual profiles, virtual machines (VMs), or rack-mounted devices.",
        "signals": ["canvas_tampering", "browser_fingerprint_collision", "coordinated_signup", "referral_cluster"],
        "falsification_criteria": [
            "unique_hardware_signature",
            "isolated_social_graph",
            "distinct_unlinked_payment_instruments",
        ],
        "allowed_tool_slots": ["SLOT_DEVICE", "SLOT_BEHAVIOR", "SLOT_GRAPH"],
        "primary_action": "quarantine_account_cluster_and_require_idv",
    },
    RiskHypothesis.SHARED: {
        "title": "Shared Network Infrastructure / False Positive (Shared Egress)",
        "is_residual": False,
        "mechanism": "Legitimate users with overlapping network attributes due to shared public Wi-Fi, mobile CGNAT IP, or office proxy.",
        "signals": ["cgnat_subnet", "diverse_user_agents_same_ip", "normal_session_duration", "legitimate_past_history"],
        "falsification_criteria": [
            "datacenter_or_commercial_vpn_detected",
            "collusion_cross_account_financial_overlap",
        ],
        "allowed_tool_slots": ["SLOT_NETWORK", "SLOT_BEHAVIOR", "SLOT_GRAPH"],
        "primary_action": "allow_with_passive_monitoring_or_soft_captcha",
    },
    RiskHypothesis.NOVEL: {
        "title": "Novel / Zero-Day",
        "is_residual": True,  # Slot thặng dư (Residual / Sink Hypothesis)
        "mechanism": "Zero-day exploit techniques, next-generation deepfakes, or business logic errors not previously recorded.",
        "signals": ["anomaly_score_spike", "unknown_parameter_injection", "unprecedented_flow_transition"],
        "falsification_criteria": [],  # STRICTLY EMPTY: Cấm tuyệt đối gán quy tắc phản bác trực tiếp lên slot thặng dư
        "allowed_tool_slots": ["SLOT_NETWORK", "SLOT_DEVICE", "SLOT_BEHAVIOR", "SLOT_GRAPH"],  # Liên kết công cụ diện rộng
        "primary_action": "shadow_friction_quarantine_and_human_dossier",
        "residual_policies": {
            "trigger_shadow_friction": True,
            "export_human_audit_dossier": True,
            "require_secops_escalation": True,
        },
    },
}


def mece_investigation_question(question_id: str = "root_cause") -> Dict[str, Any]:
    """Generate the standardized Laya decision question over the 4 MECE hypotheses.

    Each criterion is fixed to a calibrated 10-25 token description so that
    cross-entropy scoring and self-attention weights across marker positions remain
    strictly balanced without length bias.
    """
    return {
        question_id: {
            "type": "choice",
            "instructions": "Which physical root cause hypothesis best explains the behavior in `state`?",
            "criteria": {
                hyp.value: MECE_HYPOTHESIS_STRINGS[hyp]
                for hyp in RiskHypothesis
            },
        }
    }


def format_investigation_state(
    raw_payload: Union[Dict[str, Any], str],
    task_token: str = DEFAULT_TASK_TOKEN,
    context_prefix: Optional[str] = None,
) -> str:
    """Format input state prefixed with the unified task token for System 1 routing."""
    if isinstance(raw_payload, (dict, list)):
        payload_str = json.dumps(raw_payload, ensure_ascii=False)
    else:
        payload_str = str(raw_payload).strip()

    prefix = f"{context_prefix} " if context_prefix else ""
    return f"{task_token} {prefix}{payload_str}"


@dataclass
class MECEInvestigationFrame:
    """Standardized envelope for MECE hypothesis-driven investigations."""
    raw_payload: Union[Dict[str, Any], str]
    task_token: str = DEFAULT_TASK_TOKEN
    attributes: Dict[str, Any] = field(default_factory=dict)
    source: str = "AUDIT"

    def to_routing_text(self) -> str:
        return format_investigation_state(self.raw_payload, task_token=self.task_token)
