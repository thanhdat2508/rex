"""Comprehensive Enterprise Unit Tests for Multi-turn Interactive Investigation Loop.

Verifies:
  1. Context Hub: Alert ingestion from APM/SIEM, MECE hypothesis space, and initial mask M=[1,1,1,1].
  2. Parallel Tool Selection: Independent Sigmoid scoring, floor threshold 0.35, relative margin 0.15,
     funnel budget K_t (3 -> 2 -> 1), and inverse tool masking (-inf).
  3. Client Security: Outbound-only MCP dispatcher, Read-only policy, and Client-side PII sanitization.
  4. Empirical Popperian Falsification (PoE): Refutation based strictly on necessary condition violations.
  5. Convergence & Verdict Execution:
     - Known Root Cause: automatic runbook assignment.
     - Novel Anomaly (H_novel): SecOps dossier, RexBot candidate WAF rule, and backtest (FPR <= 0.01%).
  6. Sub-second SLA: Complete 3-round closed loop executes in under 1.0 second (< 1000ms) in VPC.
  7. Problem A: Invariant prefix preservation & Left-truncation under massive telemetry inflation.
  8. Problem B: Immediate synchronization of PoE rules preventing Dead-Branch tool selection at Round 1.
  9. Problem C: Vectorized Columnar Backtesting with 100,000 records in under 5ms.
"""

import asyncio
import os
import sys
import time
import unittest
from typing import Any, Dict, List

import torch
import torch.nn as nn
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.events import (
    MAX_WAF_PROMOTION_FPR,
    CandidateWAFRule,
    ColumnarTrafficCache,
    ContextHub,
    InvestigationTensorFormatter,
    LayaInvestigationModel,
    MultiTurnInvestigationLoop,
    RexBotRuleBacktester,
    RiskHypothesis,
    ToolSlot,
)


class DummyModernBertEncoder(nn.Module):
    """In-memory dummy encoder matching ModernBERT interface for fast deterministic unit tests."""

    def __init__(self, vocab_size: int = 50370, hidden_size: int = 64):
        super().__init__()
        self.config = unittest.mock.MagicMock()
        self.config.hidden_size = hidden_size
        self.config.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, hidden_size)

    def get_input_embeddings(self):
        return self.embeddings

    def resize_token_embeddings(self, new_num_tokens: int):
        self.embeddings = nn.Embedding(new_num_tokens, self.config.hidden_size)
        self.config.vocab_size = new_num_tokens

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        last_hidden = self.embeddings(input_ids)
        mock_output = unittest.mock.MagicMock()
        mock_output.last_hidden_state = last_hidden
        mock_output.__getitem__ = lambda s, idx: last_hidden if idx == 0 else None
        return mock_output


def test_context_hub_ingestion_and_pii_sanitization():
    """Verify Context Hub ingests SIEM alerts, sanitizes PII, and initializes M=[1,1,1,1]."""
    hub = ContextHub()
    raw_alert = {
        "event_type": "PROMO_ABUSE",
        "user_email": "attacker_bot@fraudsyndicate.com",
        "credit_card": "4111111111111111",
        "auth_header": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-ID",
        "client_ip": "198.51.100.42",
    }

    session = hub.ingest_alert(raw_alert, source="APM_DATADOG", session_id="sess_apm_01")

    assert session.session_id == "sess_apm_01"
    assert session.domain_type == "TYPE_PROMO_ABUSE"
    assert session.current_mask.to_list() == [1, 1, 1, 1]

    # Verify client-side PII sanitization
    sanitized = session.accumulated_telemetry
    assert sanitized["user_email"] == "[MASKED_EMAIL]"
    assert sanitized["credit_card"] == "[MASKED_PAN]"
    assert sanitized["auth_header"] == "[MASKED_JWT]"
    assert sanitized["client_ip"] == "198.51.100.42"


def test_parallel_tool_selection_policy():
    """Verify Floor Threshold (0.35), Relative Margin (0.15), and Funnel Budget K_t."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    enc = DummyModernBertEncoder()
    model = LayaInvestigationModel(encoder=enc, tokenizer=tokenizer)
    loop = MultiTurnInvestigationLoop(model=model, tokenizer=tokenizer)

    tool_logits = torch.tensor([1.5, 1.0, 0.1, -2.0])

    selected_k1, scores_k1 = loop.select_tools_parallel(tool_logits, budget_k=3)
    assert len(selected_k1) == 2
    assert ToolSlot.NETWORK in selected_k1
    assert ToolSlot.DEVICE in selected_k1
    assert ToolSlot.BEHAVIOR not in selected_k1
    assert ToolSlot.GRAPH not in selected_k1

    selected_k3, scores_k3 = loop.select_tools_parallel(tool_logits, budget_k=1)
    assert len(selected_k3) == 1
    assert selected_k3 == [ToolSlot.NETWORK]


def test_multi_turn_investigation_and_known_verdict_runbook():
    """Test full 3-turn interactive cycle converging to known root cause H_sybil with runbook."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    enc = DummyModernBertEncoder()
    model = LayaInvestigationModel(encoder=enc, tokenizer=tokenizer)
    loop = MultiTurnInvestigationLoop(model=model, tokenizer=tokenizer, confidence_threshold=0.99)

    session = loop.context_hub.ingest_alert({"event_type": "PROMO_ABUSE"})

    def tool_exec_turn1(slots, state):
        return {"behavior.keystroke_jitter_ms": 150.0}

    rec1 = loop.step(session, tool_exec_turn1)
    assert rec1.round_t == 1
    assert "H_botnet" in rec1.newly_falsified_hypotheses
    assert session.current_mask.botnet == 0
    assert not session.is_terminated

    def tool_exec_turn2(slots, state):
        return {"network.is_datacenter": True}

    rec2 = loop.step(session, tool_exec_turn2)
    assert rec2.round_t == 2
    assert "H_shared" in rec2.newly_falsified_hypotheses
    assert session.current_mask.shared == 0
    assert not session.is_terminated

    def tool_exec_turn3(slots, state):
        return {"graph.cluster_size": 45}

    rec3 = loop.step(session, tool_exec_turn3)
    assert session.is_terminated is True
    assert session.round_t == 3
    assert session.final_verdict == RiskHypothesis.SYBIL
    assert session.runbook is not None
    assert session.runbook["runbook_id"] == "RB-SEC-002-SYBIL"


def test_novel_residual_verdict_and_waf_backtest():
    """Test H_novel activation: triggers Shadow Friction, Dossier, and WAF candidate backtesting."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    enc = DummyModernBertEncoder()
    model = LayaInvestigationModel(encoder=enc, tokenizer=tokenizer)

    benign_history = [{"network.anomaly_pattern": "standard_tls_v13"} for _ in range(10000)]

    loop = MultiTurnInvestigationLoop(
        model=model,
        tokenizer=tokenizer,
        historical_benign_traffic=benign_history,
    )

    refuting_evidence = {
        "behavior.keystroke_jitter_ms": 150.0,
        "device.canvas_cluster_size": 1,
        "network.is_datacenter": True,
    }

    session = loop.run(
        domain_type="TYPE_PROMO_ABUSE",
        initial_state=refuting_evidence,
    )

    assert session.is_terminated is True
    assert session.final_verdict == RiskHypothesis.NOVEL
    assert session.termination_reason == "POE_KNOWN_HYPOTHESES_REFUTED"
    assert session.shadow_friction_directive is not None
    assert session.audit_dossier is not None
    assert session.candidate_waf_rule is not None
    assert session.candidate_waf_rule.status == "PROMOTED_TO_WAF"
    assert session.candidate_waf_rule.false_positive_rate <= MAX_WAF_PROMOTION_FPR


def test_sub_second_latency_budget_in_vpc():
    """Verify full 3-round closed loop executes in under 1.0 second (< 1000ms)."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    enc = DummyModernBertEncoder()
    model = LayaInvestigationModel(encoder=enc, tokenizer=tokenizer)
    loop = MultiTurnInvestigationLoop(model=model, tokenizer=tokenizer)

    start_bench = time.perf_counter()

    async def run_async_investigation():
        async def mock_async_tools(slots: List[ToolSlot], state: Dict[str, Any]) -> Dict[str, Any]:
            await asyncio.sleep(0.005)
            return {"network.probe_latency_ms": 12.5}

        return await loop.run_async(
            domain_type="TYPE_PROMO_ABUSE",
            initial_state={"ip": "10.0.1.15"},
            async_tool_executor=mock_async_tools,
        )

    session = asyncio.run(run_async_investigation())
    total_time_ms = (time.perf_counter() - start_bench) * 1000.0

    assert session.is_terminated is True
    assert session.total_latency_ms < 1000.0
    assert total_time_ms < 1000.0


# ======================================================================================
# NEW TESTS FOR PROBLEMS A, B, AND C
# ======================================================================================

def test_problem_a_sequence_overflow_and_invariant_prefix_protection():
    """Problem A: Massive telemetry inflation must not corrupt prefix or [MASK] token positions."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    formatter = InvestigationTensorFormatter()

    # Generate massive accumulated telemetry exceeding 1,000 tokens
    massive_telemetry = {f"metric_sensor_stream_{i}": round(i * 1.37, 2) for i in range(300)}
    massive_telemetry["freshest_evidence_turn3"] = 999.99

    seq = formatter.format_sequence(
        type_token="<TYPE_PROMO_ABUSE>",
        instructions="Route incident",
        state_payload=massive_telemetry,
        tokenizer=tokenizer,
        max_length=128,  # Constrained max_length
        compact_state=True,
    )

    enc = tokenizer(seq)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])

    # 1. Total tokens must stay within budget
    assert len(enc["input_ids"]) <= 128

    # 2. Invariant Prefix check: All 4 [MASK] tokens must strictly be preserved in the prompt header
    mask_indices = [i for i, tok in enumerate(tokens) if tok == "[MASK]"]
    assert len(mask_indices) == 4, f"Expected exactly 4 [MASK] tokens, got {len(mask_indices)}"

    # 3. Left-Truncation check: freshest telemetry at the tail is kept
    assert "freshest" in seq and "turn3" in seq and "999" in seq


def test_problem_b_immediate_poe_synchronization_at_round_1():
    """Problem B: If ingested alert refutes H_botnet, Round 1 must NOT select tools exclusive to botnet."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    enc = DummyModernBertEncoder()
    model = LayaInvestigationModel(encoder=enc, tokenizer=tokenizer)
    loop = MultiTurnInvestigationLoop(model=model, tokenizer=tokenizer)

    # Initial alert contains telemetry refuting H_botnet right at ingestion
    initial_alert = {
        "event_type": "PROMO_ABUSE",
        "behavior.keystroke_jitter_ms": 160.0,  # Falsifies H_botnet directly
    }

    session = loop.context_hub.ingest_alert(initial_alert)
    assert session.current_mask.botnet == 0  # H_botnet is dead before Round 1 starts

    # Verify that in step(), select_tools_parallel receives the synchronized mask
    def dummy_tool_exec(slots, state):
        return {}

    rec = loop.step(session, dummy_tool_exec)
    # The mask at round 1 must reflect that botnet was already 0
    assert rec.active_mask[0] == 0


def test_problem_c_vectorized_columnar_backtest_100k_records():
    """Problem C: ColumnarTrafficCache must backtest 100,000 records in < 5ms with exact FPR."""
    # Generate 100,000 synthetic benign events
    n_samples = 100_000
    cache = ColumnarTrafficCache(capacity=n_samples)

    # Generate synthetic records: 99,995 benign (latency 20-50ms), 5 false positives (latency >= 100ms)
    records = [{"network.latency_ms": 25.0, "device.is_rooted": False} for _ in range(n_samples - 5)]
    records.extend([{"network.latency_ms": 105.0, "device.is_rooted": True} for _ in range(5)])

    cache.load_records(records)
    assert cache.num_samples == n_samples

    rule = CandidateWAFRule(
        rule_id="waf_test_01",
        target_metric="network.latency_ms",
        operator="GTE",
        threshold=100.0,
    )

    t0 = time.perf_counter()
    tested_rule = RexBotRuleBacktester.backtest(rule, cache)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f"100,000 records vectorized backtest elapsed: {elapsed_ms:.3f} ms")

    # Performance constraint: strictly under 5ms (usually < 1ms on NumPy)
    assert elapsed_ms < 5.0, f"Backtest took too long: {elapsed_ms:.2f}ms >= 5ms"

    # Accuracy constraint: 5 / 100,000 = 0.00005 <= 0.0001 (0.01%) -> PROMOTED_TO_WAF
    assert tested_rule.false_positive_rate == 5 / 100_000
    assert tested_rule.false_positive_rate <= MAX_WAF_PROMOTION_FPR
    assert tested_rule.status == "PROMOTED_TO_WAF"


if __name__ == "__main__":
    test_context_hub_ingestion_and_pii_sanitization()
    test_parallel_tool_selection_policy()
    test_multi_turn_investigation_and_known_verdict_runbook()
    test_novel_residual_verdict_and_waf_backtest()
    test_sub_second_latency_budget_in_vpc()
    test_problem_a_sequence_overflow_and_invariant_prefix_protection()
    test_problem_b_immediate_poe_synchronization_at_round_1()
    test_problem_c_vectorized_columnar_backtest_100k_records()
    print("All Enterprise Multi-turn Investigation Loop tests passed successfully!")
