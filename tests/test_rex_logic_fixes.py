"""Regression tests for Rex inference/dispatch fixes found after the remediation pass.

Each test reproduces a defect that existed before the fix:
- PoE rules ignored observations grouped per tool (Part B packs {tool: {metric: value}}).
- RexAdapter re-initialized survival every round (L2 on the orchestrator path).
- hyp_scorer scored hypotheses on memory that did not yet hold the newest observations.
- The H_novel slot was assumed to be the last column, and padding slots could be revived.
- Cheap READ tools had tau = 0, so a trained gain head always called every remaining tool.
- Missing telemetry: stall_reason (L8), s_prev / rule_stall (L9), alive_hypotheses (N3).
"""
import os
import sys
import tempfile
from typing import Any, Dict, List

import torch
from transformers import AutoConfig, AutoModel, BertTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.orch.agent_port import Case
from rex.orch.rex_adapter import RexAdapter
from rex.rex_agent import InvestigatorAgent
from rex.rex_data import _domain_risk
from rex.rex_format import collate_rex, encode_rex_input, format_observations
from rex.rex_labels import generate_risk_label
from rex.rex_model import RexDecisionModel
from rex.rex_poe import ComparisonOperator, FalsificationRule, PoEFalsificationEngine
from rex.rex_registry import load_registry
from rex.rex_tools import dispatch_three_branch, gain_threshold
from rex.rex_vocab import add_type_tokens

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = ["geo_ip_lookup", "device_fingerprint", "auth_history"]
HYPS = {"H_bot": "bot", "H_sybil": "sybil", "H_novel": "novel"}


def _tok() -> BertTokenizer:
    fd, vocab_path = tempfile.mkstemp(suffix=".txt", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nwhich\nhypothesis\nbot\nsybil\nnovel\nstate\nip\nok\n")
    tok = BertTokenizer(vocab_path)
    os.remove(vocab_path)
    add_type_tokens(tok, None, {"<TYPE_PROMO>": "promo"})
    return tok


def _model() -> RexDecisionModel:
    torch.manual_seed(0)
    cfg = AutoConfig.for_model("bert", hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                               intermediate_size=64, vocab_size=200)
    return RexDecisionModel(AutoModel.from_config(cfg), tools=TOOLS).eval()


def _round1(model, tok, hypotheses=HYPS, **kw):
    enc = encode_rex_input(tok, "<TYPE_PROMO>", "which hypothesis", hypotheses, state="state")
    b = collate_rex([enc], pad_token_id=tok.pad_token_id)
    return model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], **kw)


def test_poe_rules_apply_to_observations_grouped_per_tool():
    rules = {"H_bot": [FalsificationRule("x", ComparisonOperator.GT, 0.5)], "H_novel": []}
    alive = {"H_bot": True, "H_novel": True}
    engine = PoEFalsificationEngine()
    assert engine.evaluate({"x": 0.9}, alive, rules)["H_bot"] is False
    assert engine.evaluate({"tool_a": {"x": 0.9}}, alive, rules)["H_bot"] is False


def test_adapter_keeps_rule_eliminated_hypothesis_dead_across_rounds():
    tok, model = _tok(), _model()
    rules = {"H_bot": [FalsificationRule("x", ComparisonOperator.GT, 0.5)], "H_sybil": [], "H_novel": []}
    adapter = RexAdapter(model, tok, "which hypothesis", HYPS, rules=rules, tool_specs=None)
    avail = {t: True for t in TOOLS}
    adapter.start(Case(ticket_id="t1", type_token="<TYPE_PROMO>", text="state"), avail)

    adapter.observe("geo_ip_lookup OK x=0.9", {"geo_ip_lookup": {"x": 0.9}}, avail)
    assert adapter.alive["H_bot"] is False
    adapter.observe("device_fingerprint OK y=1", {"device_fingerprint": {"y": 1}}, avail)
    assert adapter.alive["H_bot"] is False, "symbolic elimination must persist (L2)"
    assert adapter.state.round_idx == 3


def test_agent_scores_hypotheses_on_memory_holding_new_evidence():
    tok, model = _tok(), _model()
    ingests: List[int] = []          # memory length before each ingestion
    scored_after_ingest: List[int] = []  # memory length seen by the first scorer call after it
    pending = {"armed": False}
    orig_scorer, orig_ingest = model.hyp_scorer.forward, model.ingest_evidence

    def scorer(markers, M, mask=None, return_logits=False):
        if pending["armed"]:
            scored_after_ingest.append(M.size(1))
            pending["armed"] = False
        return orig_scorer(markers, M, mask, return_logits=return_logits)

    def ingest(state, *a, **kw):
        ingests.append(state.M.size(1))
        pending["armed"] = True
        return orig_ingest(state, *a, **kw)

    model.hyp_scorer.forward = scorer
    model.ingest_evidence = ingest
    agent = InvestigatorAgent(model)
    obs_by_round: List[Dict[str, Any]] = [{"ip": 0.1}, {"ip": 0.2}, {"ip": 0.3}]

    def dispatcher(tools):
        return obs_by_round[min(len(ingests), 2)]

    result = agent.investigate(
        tok=tok, type_token="<TYPE_PROMO>", instruction="which hypothesis", hypotheses=HYPS,
        initial_state="state", tool_dispatcher=dispatcher, rules={}, tool_hyp_mapping={}, max_rounds=3,
    )
    n_rounds = len(result.traces)
    assert n_rounds >= 1
    assert len(ingests) == n_rounds, "each tool round is ingested exactly once (L5)"
    assert len(scored_after_ingest) == n_rounds
    for before, scored in zip(ingests, scored_after_ingest):
        assert scored > before, "hypotheses must be scored on memory that already holds the new evidence"


def test_h_novel_slot_is_found_by_name_and_padding_stays_dead():
    tok, model = _tok(), _model()
    order = ["H_novel", "H_bot", "H_sybil"]  # shuffled layout: H_novel is not last
    with torch.no_grad():
        _, state, _ = _round1(model, tok, hyp_alive=torch.tensor([[True, True, True]]))
        alive = torch.tensor([[False, False, True]])
        _, state2, out = model(state=state, hyp_alive=alive, hyp_names=order)
    assert state2.hyp_alive.tolist() == [[True, False, True]], "H_novel kept alive by name, H_bot stays dead"
    assert out.verdict_probs[0, 1].item() < 1e-6

    # Padding slots of a batch stay masked even when a caller passes an all-True survival mask
    marker_mask = torch.tensor([[True, True, False]])
    resolved = RexDecisionModel._resolve_hyp_alive(torch.ones(1, 3, dtype=torch.bool), marker_mask, None)
    assert resolved.tolist() == [[True, True, False]]


def test_gain_threshold_and_sufficiency_first_dispatch():
    assert gain_threshold("READ", 0.0) == 0.05
    assert gain_threshold("HEAVY", 0.5) > gain_threshold("LIGHT", 0.1) > gain_threshold("READ", 0.0)
    kw = dict(tool_names=["a", "b"], tool_logits=torch.tensor([1.0, 0.5]), round_idx=2, gain_head_trained=True)

    # Enough evidence and low risk: stop instead of calling every remaining cheap tool
    assert dispatch_three_branch(tool_gains=torch.tensor([0.3, 0.2]), s_prob=torch.tensor(0.95),
                                 r_prob=torch.tensor(0.1), **kw)[:2] == ("VERDICT", [])
    # Enough evidence but high risk: more tools cannot lower risk -> STALL for a human
    assert dispatch_three_branch(tool_gains=torch.tensor([0.3, 0.2]), s_prob=torch.tensor(0.95),
                                 r_prob=torch.tensor(0.6), **kw)[:2] == ("STALL", [])
    # Not enough evidence: tools are ranked by predicted gain, those under the floor are skipped
    assert dispatch_three_branch(tool_gains=torch.tensor([0.2, 0.3]), s_prob=torch.tensor(0.2),
                                 r_prob=torch.tensor(0.1), **kw)[:2] == ("CALL_TOOL", ["b", "a"])
    assert dispatch_three_branch(tool_gains=torch.tensor([0.01, 0.3]), s_prob=torch.tensor(0.2),
                                 r_prob=torch.tensor(0.1), **kw)[:2] == ("CALL_TOOL", ["b"])


def test_step_output_carries_stall_reason_rule_stall_and_alive_names():
    tok, model = _tok(), _model()
    names = list(HYPS)
    with torch.no_grad():
        model.head_s.proj_out.bias.fill_(10.0)   # S ~ 1
        model.head_r.proj_out.bias.fill_(10.0)   # R ~ 1 -> sufficient but risky
        tool_mask = torch.zeros(1, len(TOOLS), dtype=torch.bool)  # nothing callable
        _, state, out = _round1(model, tok, hyp_names=names, tool_mask=tool_mask)
        _, _, out2 = model(state=state, hyp_names=names, tool_mask=tool_mask)
    assert out.proposed_action == ["STALL"]
    assert out.stall_reason == ["HIGH_RISK"]
    assert state.s_prev is not None and len(state.s_prev) == 1
    assert bool(out2.rule_stall[0]) is True, "S did not move between rounds, so rule_stall fires (L9)"
    trace = out.to_trace_dict(0)
    assert trace["alive_hypotheses"] == names
    assert trace["stall_reason"] == "HIGH_RISK"


def test_dispatch_flag_does_not_change_model_outputs():
    tok, model = _tok(), _model()
    with torch.no_grad():
        v1, _, o1 = _round1(model, tok, dispatch=True)
        v2, _, o2 = _round1(model, tok, dispatch=False)
    assert torch.equal(v1, v2) and torch.equal(o1.s_logit, o2.s_logit)
    assert torch.equal(o1.tool_gains, torch.sigmoid(o1.tool_gain_logits))
    assert o1.hyp_logits.shape == v1.shape
    assert o2.proposed_action == [] and o1.proposed_action != []


def test_collate_rex_returns_tool_mask():
    tok = _tok()
    enc = encode_rex_input(tok, "<TYPE_PROMO>", "which hypothesis", HYPS, state="state", tool_mask=[True, False, True])
    batch = collate_rex([enc], pad_token_id=tok.pad_token_id)
    assert batch["tool_mask"].tolist() == [[True, False, True]]


def test_risk_labels_match_registry_type_tokens_and_honor_domain_override():
    registry = load_registry(os.path.join(ROOT, "examples", "rex", "registry.example.json"))
    assert generate_risk_label("<TYPE_API_SCRAPING>") == 0.05
    domain = registry.get_domain("<TYPE_PROMO_ABUSE>")
    assert _domain_risk(domain) == 0.1
    from dataclasses import replace
    assert _domain_risk(replace(domain, risk=0.7)) == 0.7


def test_format_observations_matches_part_b_packer():
    from rex.orch.packer import pack_evidence
    from rex.orch.state_store import ToolResult

    obs = {"t2": {"b": 0.25, "a": True}, "t1": {"z": 3}}
    packed = pack_evidence([ToolResult(tool="t1", status="OK", metrics={"z": 3}),
                            ToolResult(tool="t2", status="OK", metrics={"b": 0.25, "a": True})], round_k=2)
    assert format_observations(obs) == packed.text


def test_agent_escalates_a_stall_instead_of_reporting_a_verdict():
    tok, model = _tok(), _model()
    with torch.no_grad():
        model.head_s.proj_out.bias.fill_(-10.0)  # S ~ 0: never sufficient
    result = InvestigatorAgent(model).investigate(
        tok=tok, type_token="<TYPE_PROMO>", instruction="which hypothesis", hypotheses=HYPS,
        rules={}, tool_hyp_mapping={}, initial_state="state",
        tool_dispatcher=lambda tools: {t: {"ip": 0.1} for t in tools}, max_rounds=3,
    )
    assert result.termination_reason.value in ("stall", "max_rounds_reached")
    assert result.is_escalated is True


def test_alert_metrics_falsify_hypotheses_before_round_one():
    """Rollouts apply the rules to the alert, so inference must too (train/inference parity)."""
    tok, model = _tok(), _model()
    rules = {"H_bot": [FalsificationRule("x", ComparisonOperator.GT, 0.5)], "H_sybil": [], "H_novel": []}
    seen_alive = []
    orig = model.forward

    def spy(*a, **kw):
        if kw.get("state") is None and kw.get("hyp_alive") is not None:
            seen_alive.append(kw["hyp_alive"].tolist())
        return orig(*a, **kw)

    model.forward = spy
    InvestigatorAgent(model).investigate(
        tok=tok, type_token="<TYPE_PROMO>", instruction="which hypothesis", hypotheses=HYPS, rules=rules,
        tool_hyp_mapping={}, initial_state="state", initial_observations={"x": 0.9},
        tool_dispatcher=lambda tools: {}, max_rounds=1,
    )
    assert seen_alive[0] == [[False, True, True]]

    adapter = RexAdapter(_model(), tok, "which hypothesis", HYPS, rules=rules)
    adapter.start(Case(ticket_id="t", type_token="<TYPE_PROMO>", text="state", state={"x": 0.9}),
                  {t: True for t in TOOLS})
    assert adapter.alive["H_bot"] is False


def test_agent_escalates_high_risk_automated_verdicts():
    tok, model = _tok(), _model()
    with torch.no_grad():
        model.head_r.proj_out.bias.fill_(10.0)  # R ~ 1
    rules = {"H_bot": [FalsificationRule("x", ComparisonOperator.GT, 0.5)], "H_sybil": [], "H_novel": []}
    result = InvestigatorAgent(model).investigate(
        tok=tok, type_token="<TYPE_PROMO>", instruction="which hypothesis", hypotheses=HYPS, rules=rules,
        tool_hyp_mapping={}, initial_state="state", tool_dispatcher=lambda tools: {t: {"x": 0.9} for t in tools},
        max_rounds=3,
    )
    assert result.is_escalated is True


def test_tool_mask_unmapped_cross_domain_refused():
    """Verify compute_tool_mask with allow_unmapped=False masks cross-domain tools (Fix Lỗi 2)."""
    from rex.rex_poe import HardMaskingManager
    mgr = HardMaskingManager()
    all_tools = ["geo_ip_lookup", "device_fingerprint", "payment_graph"]
    alive = {"H_promo_1": True, "H_novel": True}
    # Only geo_ip_lookup and device_fingerprint are mapped to H_promo_1. payment_graph belongs to fraud domain.
    tool_map = {
        "geo_ip_lookup": ["H_promo_1"],
        "device_fingerprint": ["H_promo_1"],
    }
    mask = mgr.compute_tool_mask(all_tools, alive, tool_map, general_tools=[], allow_unmapped=False)
    assert mask == [True, True, False], "payment_graph must be False when allow_unmapped=False"


def test_conformal_calibrator_excludes_zero_prob_hypotheses():
    """Verify predict_set excludes 0.0 probability hypotheses even when q_hat=1.0 (Fix Lỗi 3)."""
    from rex.rex_heads import ConformalCalibrator
    cal = ConformalCalibrator()
    probs = torch.tensor([[0.6, 0.4, 0.0]])
    labels = ["H_target", "H_noise", "H_eliminated"]
    sets = cal.predict_set(probs, labels, q_hat=1.0)
    assert sets[0] == ["H_target", "H_noise"]
    assert "H_eliminated" not in sets[0], "Hypothesis with 0.0 probability must not be in conformal set"


def test_trace_dict_per_sample_stall_mode():
    """Verify to_trace_dict returns sample-specific stall_mode when batching (Fix Lỗi Mới 4)."""
    from rex.rex_types import RexStepOutput
    out = RexStepOutput(
        s_logit=torch.zeros(2, 1),
        s_prob=torch.zeros(2, 1),
        r_logit=torch.zeros(2, 1),
        r_prob=torch.zeros(2, 1),
        verdict_logits=torch.zeros(2, 3),
        verdict_probs=torch.tensor([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]),
        tool_logits=torch.zeros(2, 2),
        tool_gains=torch.zeros(2, 2),
        stall=torch.tensor([True, False]),
        stall_mode="LEARNED",
        stall_modes=["LEARNED", "RULE"],
        stall_reason=["NO_GAIN", "NONE"],
        rule_stall=torch.tensor([False, True]),
        selected_tools=[[], ["tool_a"]],
        proposed_action=["STALL", "CALL_TOOL"],
        h_T=torch.zeros(2, 32),
    )
    t0 = out.to_trace_dict(batch_idx=0)
    t1 = out.to_trace_dict(batch_idx=1)
    assert t0["stall_mode"] == "LEARNED"
    assert t1["stall_mode"] == "RULE", "Sample 1 must report its own stall_mode, not stall_modes[0]"


def test_head_s_epistemic_guardrail_single_survivor():
    """Verify epistemic guardrail protects against premature convergence on single survivor (Fix Lỗi 1)."""
    tok, model = _tok(), _model()
    names = ["H_survivor", "H_novel"]
    # Single cataloged survivor. With uniform logits, survivor has prob 0.50 (high entropy/doubt).
    with torch.no_grad():
        model.head_s.proj_out.bias.fill_(10.0)  # tries to force S = 1.0
        marker_pos = torch.tensor([[5, 8]])
        marker_mask = torch.tensor([[True, True]])
        input_ids = torch.randint(0, 50, (1, 16))
        attention_mask = torch.ones((1, 16), dtype=torch.long)
        _, _, out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            marker_pos=marker_pos,
            marker_mask=marker_mask,
            hyp_names=names,
            hyp_alive=torch.tensor([[True, True]]),
        )
    # S should be constrained by the epistemic confidence, not blindly 1.0
    assert out.s_prob.item() < 0.85, f"Expected S to be constrained below 0.85, got {out.s_prob.item()}"


def test_autocast_factory_cpu_and_amp_support():
    """Verify _autocast_factory supports modern PyTorch CPU bfloat16 and cuda GradScaler (Fix Lỗi Mới 5)."""
    from rex.rex_train import _autocast_factory
    ctx_fn, scaler = _autocast_factory(torch.device("cpu"), amp=True)
    assert ctx_fn is not None
    assert scaler is None
    with ctx_fn():
        x = torch.ones(2, 2)
        y = x + x
        assert y.shape == (2, 2)

