"""Tests for the Rex trajectory trainer (T4, T10) and the label/dispatch fixes it relies on.

Runs offline with a tiny BERT encoder and a tokenizer built from the example registry vocabulary.
"""
import os
import re
import sys
import tempfile

import pytest
import torch
from transformers import AutoConfig, AutoModel, BertTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.losses import RexLossWeights
from rex.rex_data import generate, split_by_case
from rex.rex_model import RexDecisionModel, save_rex_checkpoint
from rex.rex_registry import load_registry
from rex.rex_train import RexTrainConfig, encode_case, group_cases, run_cases, train_rex
from rex.rex_vocab import add_type_tokens
from rex.train import load_checkpoint
from rex.losses import RexMultiTaskLoss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(ROOT, "examples", "rex", "registry.example.json")


@pytest.fixture(scope="module")
def registry():
    return load_registry(REGISTRY_PATH)


@pytest.fixture(scope="module")
def tok(registry):
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        words = set(re.findall(r"[a-z0-9]+|[^\sa-z0-9]", f.read().lower()))
    words |= {"choice", "question", "alert", "round", "state", "normal", "ok"}
    fd, vocab_path = tempfile.mkstemp(suffix=".txt", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(words)) + "\n")
    t = BertTokenizer(vocab_path)
    os.remove(vocab_path)
    add_type_tokens(t, None, {k: k for k in registry.domains})
    return t


def _model(tok, registry, seed=0):
    torch.manual_seed(seed)
    cfg = AutoConfig.for_model("bert", hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                               intermediate_size=64, vocab_size=len(tok))
    return RexDecisionModel(AutoModel.from_config(cfg), tools=registry.tools)


def _config(**kw):
    base = dict(epochs=1, batch_cases=4, log_every=0, seed=0, lr_heads=3e-3, lr_encoder=1e-3, warmup_ratio=0.0)
    base.update(kw)
    return RexTrainConfig(**base)


def test_encode_case_aligns_labels_with_round1_hypothesis_order(tok, registry):
    import random

    case = group_cases(generate(registry, 6, 1))[0]
    enc = encode_case(case, tok, registry.tools, registry, rng=random.Random(3))
    keys = list(registry.get_domain(case[0]["type_token"]).hypotheses_dict)
    assert len(enc.rounds) == len(case)
    assert len(enc.evidence_ids) == len(case) - 1
    for sample, labels in zip(case, enc.rounds):
        # The target index and the survival mask point at the same hypothesis in every round
        order = [k for k in keys]
        target_slot = labels["target_verdict"]
        assert labels["hyp_alive"][target_slot] is True
        assert sum(labels["hyp_alive"]) == sum(1 for k in order if sample["alive"].get(k, True))
    assert all(len(e) <= 64 for e in enc.evidence_ids)


def test_heads_stage_trains_every_head_and_only_heads(tok, registry):
    samples = generate(registry, 12, 2)
    model = _model(tok, registry)
    frozen_before = model.encoder.embeddings.word_embeddings.weight.detach().clone()
    summary = train_rex(model, tok, registry, samples, config=_config(stage="heads"), log=lambda m: None)

    norms = summary["grad_norms"]
    for name in ("head_s", "head_r", "tool_matcher", "tool_gain", "hyp_scorer"):
        assert norms[name] > 0, f"{name} received no gradient"
    for name in ("encoder", "ponder_block", "verdict_bridge", "scorer"):
        assert norms.get(name, 0.0) == 0.0, f"{name} should be frozen in the heads stage"
    assert torch.equal(model.encoder.embeddings.word_embeddings.weight, frozen_before)
    assert bool(model.gain_head_trained.item()) is True


def test_ponder_stage_reaches_ponder_block_and_verdict_bridge(tok, registry):
    samples = generate(registry, 8, 3)
    model = _model(tok, registry)
    summary = train_rex(model, tok, registry, samples, config=_config(stage="ponder"), log=lambda m: None)
    norms = summary["grad_norms"]
    assert norms["ponder_block"] > 0
    assert norms["verdict_bridge"] > 0
    assert model.verdict_bridge.g_v.abs().item() > 0, "g_v must move away from 0 (T1)"


def test_training_reduces_loss_on_a_small_set(tok, registry):
    samples = generate(registry, 8, 4)
    model = _model(tok, registry)
    summary = train_rex(model, tok, registry, samples, config=_config(stage="full", epochs=12, batch_cases=8),
                        log=lambda m: None)
    losses = summary["epoch_losses"]
    assert losses[-1] < losses[0] * 0.8, losses


def test_trajectory_path_matches_inference_path(tok, registry):
    """The trainer's round-by-round forward gives the same verdicts as a manual inference rollout."""
    case = group_cases(generate(registry, 4, 5))[0]
    model = _model(tok, registry).eval()
    enc = encode_case(case, tok, registry.tools, registry)
    loss_fn = RexMultiTaskLoss()
    with torch.no_grad():
        _, _, final = run_cases(model, [enc], tok, loss_fn, torch.device("cpu"))

        from rex.rex_data import collate_rex_train
        b = collate_rex_train([enc.round1], tok.pad_token_id)
        v, state, _ = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"],
                            hyp_alive=torch.tensor([enc.rounds[0]["hyp_alive"]]))
        for k in range(1, len(enc.rounds)):
            ev = torch.tensor([enc.evidence_ids[k - 1]])
            v, state, _ = model(state=state, evidence_new=ev, evidence_mask=torch.ones_like(ev, dtype=torch.bool),
                                hyp_alive=torch.tensor([enc.rounds[k]["hyp_alive"]]))
    assert torch.allclose(torch.softmax(v, -1)[0], final["probs"][0, : v.size(1)], atol=1e-5)


def test_checkpoint_roundtrip_keeps_gain_flag_and_conformal_threshold(tok, registry):
    samples = generate(registry, 100, 6)
    train, val = split_by_case(samples, val_ratio=0.5, seed=0)
    model = _model(tok, registry)
    summary = train_rex(model, tok, registry, train, val, config=_config(stage="heads"), log=lambda m: None)
    assert "q_hat" in summary
    # Calibration and reported evaluation use disjoint halves of the validation cases
    assert summary["calib_cases"] + summary["val_cases"] == len({s["case_id"] for s in val})
    for key in ("s_auroc", "s_auroc_within_round", "s_ece", "r_mae", "gain_auroc", "hyp_auroc",
                "stall_recall", "conformal_coverage", "conformal_mean_set_size"):
        assert key in summary["val_metrics"], key

    with tempfile.TemporaryDirectory() as d:
        save_rex_checkpoint(model, tok, d)
        loaded, _, cfg = load_checkpoint(d)
    assert cfg["arch"] == "rex2"
    assert bool(loaded.gain_head_trained.item()) is True
    assert "verdict_temperature" in summary
    assert float(loaded.verdict_temperature()) == pytest.approx(summary["verdict_temperature"], abs=1e-2)
    assert cfg["temperature"][0] == pytest.approx(summary["verdict_temperature"], abs=1e-2)
    assert loaded.conformal_threshold() == pytest.approx(summary["q_hat"], abs=1e-3)
    assert torch.allclose(loaded.tool_embeddings, model.tool_embeddings.cpu(), atol=1e-2)


def test_tool_descriptions_give_distinct_embeddings(tok, registry):
    model = _model(tok, registry)
    model.encode_tool_descriptions([registry.tool_specs[t] for t in model.tools], tok)
    emb = model.tool_embeddings
    assert not torch.allclose(emb[0], emb[1])
    assert torch.allclose(model.tool_matcher.tool_desc_embeddings,
                          torch.nn.functional.normalize(emb, dim=-1))


def test_loss_weights_without_gain_leave_gain_head_untrusted(tok, registry):
    samples = generate(registry, 4, 7)
    model = _model(tok, registry)
    train_rex(model, tok, registry, samples,
              config=_config(loss_weights=RexLossWeights(gain_weight=0.0)), log=lambda m: None)
    assert bool(model.gain_head_trained.item()) is False


def test_from_rex_starts_with_the_laya_choice_temperature(tok, registry):
    from rex.common import DecisionModel
    from rex.rex_model import from_rex

    torch.manual_seed(0)
    cfg = AutoConfig.for_model("bert", hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                               intermediate_size=64, vocab_size=len(tok))
    laya = DecisionModel(AutoModel.from_config(cfg), head_layers=1, n_act=2)
    rex = from_rex(laya, tools=registry.tools, base_cfg={"temperature": [1.7, 1.0, 1.0]})
    assert float(rex.verdict_temperature()) == pytest.approx(1.7)

    rex.eval()
    from rex.rex_data import collate_rex_train
    case = group_cases(generate(registry, 2, 9))[0]
    b = collate_rex_train([encode_case(case, tok, rex.tools, registry).round1], tok.pad_token_id)
    with torch.no_grad():
        v, _, out = rex(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"])
    assert torch.allclose(out.verdict_probs, torch.softmax(v / 1.7, -1), atol=1e-6)


def test_auroc_matches_hand_computed_values():
    from rex.rex_train import auroc

    assert auroc(torch.tensor([0.1, 0.4, 0.35, 0.8]), torch.tensor([0, 0, 1, 1])) == pytest.approx(0.75)
    assert auroc(torch.tensor([0.5, 0.5]), torch.tensor([0, 1])) == pytest.approx(0.5)
    assert auroc(torch.tensor([0.2, 0.3]), torch.tensor([1, 1])) != auroc(torch.tensor([0.2]), torch.tensor([1]))


def test_model_keeps_tool_specs_and_rejects_incomplete_ones(tok, registry):
    import warnings

    from rex.rex_registry import ToolSpec

    model = _model(tok, registry).eval()
    specs = [registry.tool_specs[t] for t in model.tools]
    model.set_tool_specs(specs)
    with tempfile.TemporaryDirectory() as d:
        save_rex_checkpoint(model, tok, d)
        loaded, _, _ = load_checkpoint(d)
    assert loaded.tool_meta == model.tool_meta

    case = group_cases(generate(registry, 2, 1))[0]
    from rex.rex_data import collate_rex_train
    b = collate_rex_train([encode_case(case, tok, model.tools, registry).round1], tok.pad_token_id)
    args = (b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"])
    with pytest.raises(ValueError, match="missing model tools"):
        model(*args, tool_specs=[ToolSpec(name=model.tools[0], returns=[])])
    bare = _model(tok, registry).eval()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bare(*args)
    assert any("no ToolSpec" in str(w.message) for w in caught)


def test_laya_train_loop_warns_that_it_trains_only_the_verdict_path(tok, registry):
    from rex.train import TrainConfig, items_from_rows, train_model

    row = {"state": "alert state normal",
           "questions": {"q": {"type": "choice", "instructions": "which hypothesis", "criteria": {"a": "botnet", "b": "sybil"}}},
           "gold": {"q": {"probabilities": {"a": 0.9, "b": 0.1}}}}
    items, _ = items_from_rows(tok, [row] * 2, 64, 40)
    model = _model(tok, registry)
    with pytest.warns(RuntimeWarning, match="trains only the verdict path"):
        train_model(model, tok, items, TrainConfig(epochs=1, log_every=0), torch.device("cpu"), 64, 40)


def test_risk_is_the_verdict_weighted_hypothesis_risk_and_varies_by_cause(tok, registry):
    from rex.rex_data import case_risk, collate_rex_train

    dom = registry.get_domain("<TYPE_ACCOUNT_TAKEOVER>")
    assert case_risk(dom, "H_legitimate_travel") < 0.3 < case_risk(dom, "H_session_hijack")

    case = group_cases(generate(registry, 5, 2))[0]
    enc = encode_case(case, tok, registry.tools, registry)
    assert all(len(r["hyp_risk"]) == len(r["hyp_alive"]) for r in enc.rounds)
    model = _model(tok, registry).eval()
    b = collate_rex_train([enc.round1], tok.pad_token_id)
    with torch.no_grad():
        _, _, out = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"])
    assert torch.allclose(out.r_prob[:, 0], (out.verdict_probs * out.hyp_risk).sum(-1), atol=1e-5)

    steps = []
    _, parts, _ = run_cases(model, [enc], tok, RexMultiTaskLoss(), torch.device("cpu"), collect=steps)
    assert parts["hyp_risk_loss"] > 0


def test_checkpoints_without_the_survivor_count_embedding_still_load(tok, registry):
    model = _model(tok, registry)
    assert torch.count_nonzero(model.alive_count_emb.weight) == 0, "zero-init keeps untrained outputs unchanged"
    old = {k: v for k, v in model.state_dict().items() if not k.startswith("alive_count_emb.")}
    fresh = _model(tok, registry, seed=1)
    fresh.load_state_dict(old, strict=True)
    assert torch.count_nonzero(fresh.alive_count_emb.weight) == 0
