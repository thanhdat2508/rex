"""Multi-Round Investigation Trajectory Simulation, Data Synthesis Engine, and PyTorch DataLoader.

Implements Step 3 of the Falsification & MECE Architecture:
  1. Multi-round investigation trajectories (t in {1, 2, 3}) simulating the Process of Elimination (PoE).
  2. Funnel Budget Enforcement (K_t):
     - Round 1 (K_1 = 3): 2-3 exploratory tools.
     - Round 2 (K_2 = 2): 1-2 focused tools.
     - Round 3 (K_3 = 1): exactly 1 decisive verification tool.
  3. Controlled Novel/Zero-day Mixture: exactly 10% - 15% (default 12.5%) H_novel anomalies,
     where evidence falsifies all known hypotheses (M_0..M_2 = 0), forcing verdict weight to H_novel.
  4. Production-grade InvestigationDataset, InvestigationCollator, and DataLoader.
"""

from dataclasses import dataclass
import random
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .backbone import (
    CANONICAL_HYPOTHESES,
    InvestigationTensorFormatter,
)
from .mece import DEFAULT_TASK_TOKEN, RiskHypothesis
from .types import EventType


@dataclass
class InvestigationSample:
    """A single decision turn t within a multi-round investigation trajectory."""
    trajectory_id: str
    round_t: int                    # 1, 2, or 3
    type_token: str                 # e.g. "<TYPE_CREDENTIAL_STUFFING>", "<TYPE_PROMO_ABUSE>"
    instructions: str               # Investigation prompt
    state_payload: Dict[str, Any]   # Accumulated telemetry state at round t
    poe_mask: List[int]             # Hypothesis mask [M_botnet, M_sybil, M_shared, M_novel] in {0, 1}^4
    tool_labels: List[float]        # Multi-hot targets in {0.0, 1.0}^4 [NET, DEV, BEH, GRP]
    target_verdict: int             # Ground truth hypothesis index (0..3 in CANONICAL_HYPOTHESES)
    target_hypothesis: RiskHypothesis

    def to_input_dict(self) -> Dict[str, Any]:
        """Convert to dict compatible with InvestigationTensorFormatter."""
        return {
            "type_token": self.type_token,
            "instructions": self.instructions,
            "state_payload": self.state_payload,
        }



class TrajectorySynthesisEngine:
    """Generates synthetic, physically consistent multi-round investigation trajectories.

    Simulates the Popperian Process of Elimination (PoE) across rounds t=1..3 with
    funnel budget constraints K_1=3, K_2=2, K_3=1, and an explicit 10%-15% H_novel distribution.
    """

    def __init__(
        self,
        novel_ratio: float = 0.125,  # 12.5% strictly within [10%, 15%]
        seed: Optional[int] = 42,
    ):
        if not (0.10 <= novel_ratio <= 0.15):
            raise ValueError(f"novel_ratio={novel_ratio} must be strictly within [0.10, 0.15]")
        self.novel_ratio = novel_ratio
        self.rng = random.Random(seed)

    def _generate_botnet_trajectory(self, traj_id: str) -> List[InvestigationSample]:
        """Generate a 3-turn trajectory where ground truth is H_botnet."""
        event_type = self.rng.choice([
            EventType.CREDENTIAL_STUFFING.value,
            EventType.API_SCRAPING.value,
            EventType.ACCOUNT_TAKEOVER.value,
        ])
        ip = f"198.51.100.{self.rng.randint(10, 250)}"
        target_idx = CANONICAL_HYPOTHESES.index(RiskHypothesis.BOTNET)

        # Round 1 (t=1, K_1=3 exploratory tools: NETWORK, DEVICE, BEHAVIOR)
        s1 = {
            "ip": ip,
            "req_rate_rps": self.rng.randint(45, 180),
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "failed_attempts": self.rng.randint(120, 600),
        }
        # Multi-hot tool target for round 1: [NET=1, DEV=1, BEH=1, GRP=0]
        tools_t1 = [1.0, 1.0, 1.0, 0.0]
        mask_t1 = [1, 1, 1, 1]  # All active initially

        sample_t1 = InvestigationSample(
            trajectory_id=traj_id,
            round_t=1,
            type_token=event_type,
            instructions="Identify root cause hypothesis and route exploratory investigation",
            state_payload=s1,
            poe_mask=mask_t1,
            tool_labels=tools_t1,
            target_verdict=target_idx,
            target_hypothesis=RiskHypothesis.BOTNET,
        )

        # Round 2 (t=2, K_2=2 focused tools: DEVICE, BEHAVIOR)
        # Evidence collected from round 1: Datacenter IP detected -> Falsifies H_shared (not CGNAT/public WiFi)
        s2 = dict(s1)
        s2.update({
            "t1_net_asn": "AS16509_AMAZON_DATACENTER",
            "t1_net_is_datacenter": True,
            "t1_beh_mouse_curvature": 0.04,  # highly linear / artificial
            "t1_beh_keystroke_jitter_ms": 0.2,
        })
        tools_t2 = [0.0, 1.0, 1.0, 0.0]
        mask_t2 = [1, 1, 0, 1]  # H_shared eliminated

        sample_t2 = InvestigationSample(
            trajectory_id=traj_id,
            round_t=2,
            type_token=event_type,
            instructions="Collect falsification evidence to isolate automated botnet vs sybil",
            state_payload=s2,
            poe_mask=mask_t2,
            tool_labels=tools_t2,
            target_verdict=target_idx,
            target_hypothesis=RiskHypothesis.BOTNET,
        )

        # Round 3 (t=3, K_3=1 decisive tool: DEVICE or BEHAVIOR)
        # Evidence from round 2: Headless browser flags found -> Falsifies H_sybil (not human syndicate)
        s3 = dict(s2)
        s3.update({
            "t2_dev_webdriver": True,
            "t2_dev_headless_artifacts": ["cdc_adoQpoasnfa76pfcZLmcfl", "navigator.plugins_empty"],
            "t2_beh_touch_events": 0,
        })
        tools_t3 = [0.0, 1.0, 0.0, 0.0]  # Only DEVICE verification needed
        mask_t3 = [1, 0, 0, 0]  # Only H_botnet survives

        sample_t3 = InvestigationSample(
            trajectory_id=traj_id,
            round_t=3,
            type_token=event_type,
            instructions="Verify definitive device automation signature for final verdict",
            state_payload=s3,
            poe_mask=mask_t3,
            tool_labels=tools_t3,
            target_verdict=target_idx,
            target_hypothesis=RiskHypothesis.BOTNET,
        )

        return [sample_t1, sample_t2, sample_t3]

    def _generate_sybil_trajectory(self, traj_id: str) -> List[InvestigationSample]:
        """Generate a 3-turn trajectory where ground truth is H_sybil."""
        event_type = self.rng.choice([
            EventType.PROMO_ABUSE.value,
            EventType.PAYMENT_FRAUD.value,
            DEFAULT_TASK_TOKEN,
        ])
        target_idx = CANONICAL_HYPOTHESES.index(RiskHypothesis.SYBIL)

        # Round 1 (t=1, K_1=3 exploratory tools: DEVICE, BEHAVIOR, GRAPH)
        s1 = {
            "account_created_sec": self.rng.randint(60, 300),
            "promo_code": "BONUS_CLAIM_50",
            "claimed_amount": self.rng.randint(50, 500),
            "claimed_velocity": self.rng.randint(8, 30),
        }
        tools_t1 = [0.0, 1.0, 1.0, 1.0]  # Sybil does not primarily depend on single IP
        mask_t1 = [1, 1, 1, 1]

        sample_t1 = InvestigationSample(
            trajectory_id=traj_id,
            round_t=1,
            type_token=event_type,
            instructions="Identify root cause hypothesis and route exploratory investigation",
            state_payload=s1,
            poe_mask=mask_t1,
            tool_labels=tools_t1,
            target_verdict=target_idx,
            target_hypothesis=RiskHypothesis.SYBIL,
        )

        # Round 2 (t=2, K_2=2 focused tools: GRAPH, DEVICE)
        # Evidence from round 1: Human mouse curves detected -> Falsifies H_botnet (human operator)
        s2 = dict(s1)
        s2.update({
            "t1_beh_mouse_entropy": 0.88,  # Organic human curves
            "t1_beh_keystroke_delays_human": True,
            "t1_dev_canvas_entropy": 14.2,  # Virtualized or anti-detect browser spoof
        })
        tools_t2 = [0.0, 1.0, 0.0, 1.0]  # [DEV=1, GRP=1]
        mask_t2 = [0, 1, 1, 1]  # H_botnet eliminated

        sample_t2 = InvestigationSample(
            trajectory_id=traj_id,
            round_t=2,
            type_token=event_type,
            instructions="Collect relationship graph and device cluster evidence for sybil ring",
            state_payload=s2,
            poe_mask=mask_t2,
            tool_labels=tools_t2,
            target_verdict=target_idx,
            target_hypothesis=RiskHypothesis.SYBIL,
        )

        # Round 3 (t=3, K_3=1 decisive tool: GRAPH)
        # Evidence from round 2: Shared payout instrument & multi-account cluster found -> Falsifies H_shared
        s3 = dict(s2)
        s3.update({
            "t2_grp_shared_payout_hash": "hash_bank_cc901",
            "t2_grp_cluster_size": self.rng.randint(15, 60),
            "t2_dev_canvas_hash_duplicate": True,
        })
        tools_t3 = [0.0, 0.0, 0.0, 1.0]  # Only GRAPH needed
        mask_t3 = [0, 1, 0, 0]  # Only H_sybil survives

        sample_t3 = InvestigationSample(
            trajectory_id=traj_id,
            round_t=3,
            type_token=event_type,
            instructions="Verify syndicate identity graph link for final verdict",
            state_payload=s3,
            poe_mask=mask_t3,
            tool_labels=tools_t3,
            target_verdict=target_idx,
            target_hypothesis=RiskHypothesis.SYBIL,
        )

        return [sample_t1, sample_t2, sample_t3]

    def _generate_shared_trajectory(self, traj_id: str) -> List[InvestigationSample]:
        """Generate a 3-turn trajectory where ground truth is H_shared (Legitimate CGNAT/Public WiFi)."""
        event_type = self.rng.choice([
            EventType.BENIGN.value,
            EventType.ACCOUNT_TAKEOVER.value,
            DEFAULT_TASK_TOKEN,
        ])
        target_idx = CANONICAL_HYPOTHESES.index(RiskHypothesis.SHARED)

        # Round 1 (t=1, K_1=3 exploratory tools: NETWORK, BEHAVIOR, GRAPH)
        s1 = {
            "ip": f"100.64.{self.rng.randint(1, 250)}.{self.rng.randint(1, 250)}",
            "concurrent_users_on_ip": self.rng.randint(80, 450),
            "auth_success_ratio": 0.95,
        }
        tools_t1 = [1.0, 0.0, 1.0, 1.0]
        mask_t1 = [1, 1, 1, 1]

        sample_t1 = InvestigationSample(
            trajectory_id=traj_id,
            round_t=1,
            type_token=event_type,
            instructions="Identify root cause hypothesis and route exploratory investigation",
            state_payload=s1,
            poe_mask=mask_t1,
            tool_labels=tools_t1,
            target_verdict=target_idx,
            target_hypothesis=RiskHypothesis.SHARED,
        )

        # Round 2 (t=2, K_2=2 focused tools: NETWORK, GRAPH)
        # Evidence from round 1: Diverse organic browsing -> Falsifies H_botnet
        s2 = dict(s1)
        s2.update({
            "t1_beh_mouse_curvature_entropy": 0.96,
            "t1_beh_touch_variance": "high_organic_variance",
        })
        tools_t2 = [1.0, 0.0, 0.0, 1.0]  # [NET=1, GRP=1]
        mask_t2 = [0, 1, 1, 1]

        sample_t2 = InvestigationSample(
            trajectory_id=traj_id,
            round_t=2,
            type_token=event_type,
            instructions="Collect carrier routing and historical graph isolation evidence",
            state_payload=s2,
            poe_mask=mask_t2,
            tool_labels=tools_t2,
            target_verdict=target_idx,
            target_hypothesis=RiskHypothesis.SHARED,
        )

        # Round 3 (t=3, K_3=1 decisive tool: NETWORK)
        # Evidence from round 2: Telco carrier CGNAT pool confirmed, zero graph overlap -> Falsifies H_sybil
        s3 = dict(s2)
        s3.update({
            "t2_net_cgnat_gateway": True,
            "t2_net_isp": "Viettel_Mobile_CGNAT_Pool",
            "t2_grp_account_overlap": 0.0,  # Completely independent users over years
        })
        tools_t3 = [1.0, 0.0, 0.0, 0.0]  # Only NETWORK needed
        mask_t3 = [0, 0, 1, 0]  # Only H_shared survives

        sample_t3 = InvestigationSample(
            trajectory_id=traj_id,
            round_t=3,
            type_token=event_type,
            instructions="Verify carrier-grade NAT gateway allocation for final verdict",
            state_payload=s3,
            poe_mask=mask_t3,
            tool_labels=tools_t3,
            target_verdict=target_idx,
            target_hypothesis=RiskHypothesis.SHARED,
        )

        return [sample_t1, sample_t2, sample_t3]

    def _generate_novel_trajectory(self, traj_id: str) -> List[InvestigationSample]:
        """Generate a 3-turn trajectory where ground truth is H_novel (Zero-day / Logic Exploit).

        All traditional hypotheses (botnet, sybil, shared) are falsified step-by-step by evidence,
        leaving M_0..M_2 = 0 and M_novel = 1, forcing full verdict probability to H_novel.
        """
        event_type = self.rng.choice([
            EventType.UNKNOWN.value,
            EventType.PAYMENT_FRAUD.value,
            DEFAULT_TASK_TOKEN,
        ])
        target_idx = CANONICAL_HYPOTHESES.index(RiskHypothesis.NOVEL)

        # Round 1 (t=1, K_1=3 exploratory tools: NETWORK, DEVICE, GRAPH)
        s1 = {
            "unusual_api_payload": "negative_cart_quantity_bypass",
            "discrepancy_amount_usd": self.rng.randint(2500, 15000),
            "account_age_days": self.rng.randint(400, 1200),  # Long-standing VIP user!
        }
        tools_t1 = [1.0, 1.0, 0.0, 1.0]  # [NET, DEV, GRP]
        mask_t1 = [1, 1, 1, 1]

        sample_t1 = InvestigationSample(
            trajectory_id=traj_id,
            round_t=1,
            type_token=event_type,
            instructions="Identify root cause hypothesis for uncataloged transaction anomaly",
            state_payload=s1,
            poe_mask=mask_t1,
            tool_labels=tools_t1,
            target_verdict=target_idx,
            target_hypothesis=RiskHypothesis.NOVEL,
        )

        # Round 2 (t=2, K_2=2 focused tools: DEVICE, GRAPH)
        # Evidence from round 1: Clean residential IP, non-automated -> Falsifies H_botnet
        s2 = dict(s1)
        s2.update({
            "t1_net_ip_reputation": "clean_residential_fios",
            "t1_dev_hardware_integrity": "authentic_macbook_pro_m3",
            "t1_beh_organic_touch": True,
        })
        tools_t2 = [0.0, 1.0, 0.0, 1.0]  # [DEV, GRP]
        mask_t2 = [0, 1, 1, 1]  # H_botnet eliminated

        sample_t2 = InvestigationSample(
            trajectory_id=traj_id,
            round_t=2,
            type_token=event_type,
            instructions="Collect cross-layer telemetry to falsify syndicate and shared infrastructure",
            state_payload=s2,
            poe_mask=mask_t2,
            tool_labels=tools_t2,
            target_verdict=target_idx,
            target_hypothesis=RiskHypothesis.NOVEL,
        )

        # Round 3 (t=3, K_3=1 decisive tool: GRAPH)
        # Evidence from round 2: Single unique account, no CGNAT, no syndicate -> Falsifies H_sybil and H_shared
        s3 = dict(s2)
        s3.update({
            "t2_grp_syndicate_cluster": False,
            "t2_grp_device_clones": 0,
            "t2_net_cgnat": False,
            "t2_logic_exploit_signature": "race_condition_balance_double_spend",
        })
        tools_t3 = [0.0, 0.0, 0.0, 1.0]  # [GRAPH]
        # ALL 3 traditional hypotheses FALSIFIED: M_botnet=0, M_sybil=0, M_shared=0!
        mask_t3 = [0, 0, 0, 1]  # Only H_novel survives!

        sample_t3 = InvestigationSample(
            trajectory_id=traj_id,
            round_t=3,
            type_token=event_type,
            instructions="Confirm zero-day business logic vulnerability signature for final verdict",
            state_payload=s3,
            poe_mask=mask_t3,
            tool_labels=tools_t3,
            target_verdict=target_idx,
            target_hypothesis=RiskHypothesis.NOVEL,
        )

        return [sample_t1, sample_t2, sample_t3]

    def generate_trajectories(self, num_trajectories: int = 100) -> List[List[InvestigationSample]]:
        """Generate a specified number of 3-round investigation trajectories.

        Guarantees that H_novel represents between 10% and 15% (default 12.5%) of the dataset.
        """
        num_novel = max(1, int(round(num_trajectories * self.novel_ratio)))
        remaining = num_trajectories - num_novel
        # Distribute remaining evenly across the other 3 MECE hypotheses
        num_botnet = remaining // 3
        num_sybil = remaining // 3
        num_shared = remaining - (num_botnet + num_sybil)

        trajectories: List[List[InvestigationSample]] = []
        traj_counter = 0

        # 1. H_novel trajectories (10%-15%)
        for _ in range(num_novel):
            traj_counter += 1
            trajectories.append(self._generate_novel_trajectory(f"traj_novel_{traj_counter:05d}"))

        # 2. H_botnet trajectories
        for _ in range(num_botnet):
            traj_counter += 1
            trajectories.append(self._generate_botnet_trajectory(f"traj_botnet_{traj_counter:05d}"))

        # 3. H_sybil trajectories
        for _ in range(num_sybil):
            traj_counter += 1
            trajectories.append(self._generate_sybil_trajectory(f"traj_sybil_{traj_counter:05d}"))

        # 4. H_shared trajectories
        for _ in range(num_shared):
            traj_counter += 1
            trajectories.append(self._generate_shared_trajectory(f"traj_shared_{traj_counter:05d}"))

        self.rng.shuffle(trajectories)
        return trajectories

    def generate_flat_samples(self, num_trajectories: int = 100) -> List[InvestigationSample]:
        """Generate all turns flattened across all trajectories (total samples = 3 * num_trajectories)."""
        trajectories = self.generate_trajectories(num_trajectories=num_trajectories)
        flat = [turn for traj in trajectories for turn in traj]
        self.rng.shuffle(flat)
        return flat

class InvestigationDataset(Dataset):
    """PyTorch Dataset wrapping multi-round investigation turns."""

    def __init__(self, samples: List[InvestigationSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> InvestigationSample:
        return self.samples[idx]


class InvestigationCollator:
    """Collates a batch of InvestigationSamples into model-ready PyTorch tensors."""

    def __init__(
        self,
        tokenizer: Any,
        formatter: Optional[InvestigationTensorFormatter] = None,
        max_length: int = 512,
        padding: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.tokenizer = tokenizer
        self.formatter = formatter or InvestigationTensorFormatter()
        self.max_length = max_length
        self.padding = padding
        self.device = device

    def __call__(self, batch: List[InvestigationSample]) -> Dict[str, torch.Tensor]:
        """Collate and tokenize batch of InvestigationSamples."""
        # 1. Format and encode token sequences (guaranteeing state_t-only truncation)
        input_dicts = [s.to_input_dict() for s in batch]
        encoded_batch = self.formatter.encode_batch(
            tokenizer=self.tokenizer,
            items=input_dicts,
            max_length=self.max_length,
            padding=self.padding,
            device=self.device,
        )

        # 2. Extract supervisory labels
        tool_labels_list = [s.tool_labels for s in batch]
        poe_mask_list = [s.poe_mask for s in batch]
        verdict_list = [s.target_verdict for s in batch]
        round_list = [s.round_t for s in batch]

        tool_labels = torch.tensor(tool_labels_list, dtype=torch.float32)
        poe_mask = torch.tensor(poe_mask_list, dtype=torch.float32)
        target_verdict = torch.tensor(verdict_list, dtype=torch.long)
        round_t = torch.tensor(round_list, dtype=torch.long)

        if self.device is not None:
            tool_labels = tool_labels.to(self.device)
            poe_mask = poe_mask.to(self.device)
            target_verdict = target_verdict.to(self.device)
            round_t = round_t.to(self.device)

        encoded_batch.update({
            "tool_labels": tool_labels,        # [B, 4] multi-hot for Asymmetric Focal Loss
            "poe_mask": poe_mask,              # [B, 4] binary vector for Process of Elimination
            "target_verdict": target_verdict,  # [B] class index 0..3 for Verdict Head loss
            "round_t": round_t,                # [B] investigation turn index
        })

        return encoded_batch


def create_investigation_dataloader(
    samples: List[InvestigationSample],
    tokenizer: Any,
    batch_size: int = 16,
    shuffle: bool = True,
    max_length: int = 512,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
) -> DataLoader:
    """Convenience constructor creating a production-ready PyTorch DataLoader."""
    dataset = InvestigationDataset(samples)
    collator = InvestigationCollator(
        tokenizer=tokenizer,
        max_length=max_length,
        device=device,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
    )
