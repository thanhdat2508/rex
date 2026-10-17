"""Unit tests for Step 3: Multi-Round Investigation Trajectory Simulation,
Funnel Budget Enforcement, Novel Mixture Control, and Custom DataLoader.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer, BertConfig, BertModel

from rex.events import (
    CANONICAL_HYPOTHESES,
    InvestigationDataset,
    LayaInvestigationModel,
    RiskHypothesis,
    TrajectorySynthesisEngine,
    create_investigation_dataloader,
    setup_investigation_tokenizer,
)


def test_trajectory_structure_and_funnel_budget():
    """Verify that each trajectory consists of turns t=1..3 and strictly obeys
    the funnel budget constraints K_1=3, K_2=2, K_3=1.
    """
    engine = TrajectorySynthesisEngine(novel_ratio=0.125, seed=42)
    trajectories = engine.generate_trajectories(num_trajectories=40)

    assert len(trajectories) == 40

    for traj in trajectories:
        assert len(traj) == 3, "Each trajectory must contain exactly 3 turns"

        # Check turn 1
        t1 = traj[0]
        assert t1.round_t == 1
        assert sum(t1.poe_mask) == 4, "Round 1 must have all 4 hypotheses active"
        assert 2 <= sum(t1.tool_labels) <= 3, f"Round 1 K_1=3 constraint failed: sum={sum(t1.tool_labels)}"

        # Check turn 2
        t2 = traj[1]
        assert t2.round_t == 2
        assert sum(t2.poe_mask) <= 3, "Round 2 must have eliminated at least 1 hypothesis via PoE"
        assert 1 <= sum(t2.tool_labels) <= 2, f"Round 2 K_2=2 constraint failed: sum={sum(t2.tool_labels)}"

        # Check turn 3
        t3 = traj[2]
        assert t3.round_t == 3
        assert sum(t3.tool_labels) == 1, f"Round 3 K_3=1 constraint failed: sum={sum(t3.tool_labels)}"
        # Ground truth hypothesis must remain active (M_target == 1)
        assert t3.poe_mask[t3.target_verdict] == 1


def test_novel_hypothesis_anomaly_properties():
    """Verify that for H_novel trajectories:
    1. At round 3, all known hypotheses are eliminated: M_0=0, M_1=0, M_2=0 and M_novel=1.
    2. Target verdict is exactly the index of H_novel (index 3).
    """
    engine = TrajectorySynthesisEngine(novel_ratio=0.125, seed=42)
    trajectories = engine.generate_trajectories(num_trajectories=40)

    novel_idx = CANONICAL_HYPOTHESES.index(RiskHypothesis.NOVEL)
    novel_trajs = [traj for traj in trajectories if traj[0].target_verdict == novel_idx]

    assert len(novel_trajs) > 0, "Must contain at least 1 H_novel trajectory"

    for traj in novel_trajs:
        t3 = traj[2]
        # At round 3, M_botnet=0, M_sybil=0, M_shared=0 and M_novel=1
        assert t3.poe_mask == [0, 0, 0, 1], f"H_novel round 3 mask should be [0,0,0,1], got {t3.poe_mask}"
        assert t3.target_verdict == novel_idx
        assert t3.target_hypothesis == RiskHypothesis.NOVEL


def test_novel_ratio_distribution():
    """Verify H_novel represents between 10% and 15% of total generated trajectories."""
    engine = TrajectorySynthesisEngine(novel_ratio=0.125, seed=123)
    num_trajectories = 200
    trajectories = engine.generate_trajectories(num_trajectories=num_trajectories)

    novel_idx = CANONICAL_HYPOTHESES.index(RiskHypothesis.NOVEL)
    novel_count = sum(1 for traj in trajectories if traj[0].target_verdict == novel_idx)
    actual_ratio = novel_count / num_trajectories

    assert 0.10 <= actual_ratio <= 0.15, f"H_novel ratio {actual_ratio:.3f} outside [0.10, 0.15]"


def test_custom_dataset_and_dataloader():
    """Verify Custom Dataset and DataLoader yield properly batched tensors for multi-task training."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    setup_investigation_tokenizer(tokenizer)

    engine = TrajectorySynthesisEngine(novel_ratio=0.125, seed=42)
    samples = engine.generate_flat_samples(num_trajectories=16)  # 16 * 3 = 48 turn samples

    dataset = InvestigationDataset(samples)
    assert len(dataset) == 48

    dataloader = create_investigation_dataloader(
        samples=samples,
        tokenizer=tokenizer,
        batch_size=8,
        shuffle=True,
        max_length=256,
    )

    batch = next(iter(dataloader))

    # 1. Check tensor presence and shapes
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "cls_indices" in batch
    assert "mask_indices" in batch
    assert "tool_labels" in batch
    assert "poe_mask" in batch
    assert "target_verdict" in batch
    assert "round_t" in batch

    assert batch["input_ids"].shape[0] == 8
    assert batch["cls_indices"].shape == (8,)
    assert batch["mask_indices"].shape == (8, 4)
    assert batch["tool_labels"].shape == (8, 4)
    assert batch["poe_mask"].shape == (8, 4)
    assert batch["target_verdict"].shape == (8,)
    assert batch["round_t"].shape == (8,)

    # 2. Check types
    assert batch["tool_labels"].dtype == torch.float32
    assert batch["poe_mask"].dtype == torch.float32
    assert batch["target_verdict"].dtype == torch.long
    assert batch["round_t"].dtype == torch.long

    # 3. Check values in valid ranges
    assert (batch["target_verdict"] >= 0).all() and (batch["target_verdict"] <= 3).all()
    assert (batch["round_t"] >= 1).all() and (batch["round_t"] <= 3).all()


def test_dataloader_batch_feeds_into_laya_model():
    """Verify that a batch straight from DataLoader feeds into LayaInvestigationModel
    and produces unmasked raw logits for loss calculation.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    setup_investigation_tokenizer(tokenizer)

    cfg = BertConfig(
        vocab_size=len(tokenizer),
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
    )
    encoder = BertModel(cfg)
    model = LayaInvestigationModel(encoder=encoder, tokenizer=tokenizer)
    model.train()

    engine = TrajectorySynthesisEngine(novel_ratio=0.125, seed=42)
    samples = engine.generate_flat_samples(num_trajectories=4)

    dataloader = create_investigation_dataloader(
        samples=samples,
        tokenizer=tokenizer,
        batch_size=4,
        max_length=256,
    )
    batch = next(iter(dataloader))

    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        cls_indices=batch["cls_indices"],
        mask_indices=batch["mask_indices"],
    )

    # Check that tool_logits and verdict_logits match label dimensions
    assert out.raw_tool_logits.shape == batch["tool_labels"].shape  # [4, 4]
    assert out.verdict_logits.shape == (4, 4)

    # Multi-task loss computation (BCE on tool logits + CrossEntropy on verdict logits)
    bce_loss_fn = torch.nn.BCEWithLogitsLoss()
    ce_loss_fn = torch.nn.CrossEntropyLoss()

    loss_tools = bce_loss_fn(out.raw_tool_logits, batch["tool_labels"])
    loss_verdict = ce_loss_fn(out.verdict_logits, batch["target_verdict"])
    total_loss = loss_tools + loss_verdict

    total_loss.backward()

    assert model.tool_head.fc1.weight.grad is not None
    assert model.verdict_head.proj.weight.grad is not None


if __name__ == "__main__":
    test_trajectory_structure_and_funnel_budget()
    test_novel_hypothesis_anomaly_properties()
    test_novel_ratio_distribution()
    test_custom_dataset_and_dataloader()
    test_dataloader_batch_feeds_into_laya_model()
    print("All Step 3 Data Synthesis tests passed successfully!")
