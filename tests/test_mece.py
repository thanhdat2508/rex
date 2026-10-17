"""Tests for the MECE Risk Hypothesis Framework and unified task token."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer

from rex.events import (
    DEFAULT_TASK_TOKEN,
    MECE_HYPOTHESIS_STRINGS,
    RiskHypothesis,
    TASK_TOKEN_AUDIT,
    TASK_TOKEN_INVESTIGATION,
    format_investigation_state,
    mece_investigation_question,
    register_task_token,
    verify_mece_hypotheses_lengths,
)


def test_mece_hypothesis_token_lengths():
    """Verify all 4 hypothesis strings are strictly between 10 and 25 tokens."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    res = verify_mece_hypotheses_lengths(tokenizer, min_tokens=10, max_tokens=25)

    assert res["all_valid"] is True, f"Length violations found: {res['violations']}"
    for hyp, length in res["lengths"].items():
        assert 10 <= length <= 25, f"Hypothesis {hyp} has length {length}, expected 10-25 tokens"


def test_single_task_token_registration():
    """Verify registering a single task token produces a single non-split token."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    stats = register_task_token(tokenizer, task_token=TASK_TOKEN_INVESTIGATION)

    assert stats["is_valid"] is True
    assert stats["added_count"] >= 1

    encoded = tokenizer.encode(TASK_TOKEN_INVESTIGATION, add_special_tokens=False)
    assert len(encoded) == 1, f"Expected 1 token, got {encoded}"

    # Verify alias token as well
    encoded_audit = tokenizer.encode(TASK_TOKEN_AUDIT, add_special_tokens=False)
    assert len(encoded_audit) == 1, f"Expected 1 token, got {encoded_audit}"


def test_mece_question_structure():
    """Verify MECE question dictionary conforms to Laya choice question specification."""
    q = mece_investigation_question()
    assert "root_cause" in q
    spec = q["root_cause"]
    assert spec["type"] == "choice"
    criteria = spec["criteria"]
    assert len(criteria) == 4
    for hyp in RiskHypothesis:
        assert hyp.value in criteria
        assert criteria[hyp.value] == MECE_HYPOTHESIS_STRINGS[hyp]


def test_format_investigation_state():
    """Verify formatting raw payload with the task token."""
    payload = {"ip": "1.2.3.4", "failed_logins": 100}
    state = format_investigation_state(payload, task_token=DEFAULT_TASK_TOKEN)
    assert state.startswith("<TASK_RISK_INVESTIGATION>")
    assert '"ip": "1.2.3.4"' in state


if __name__ == "__main__":
    test_mece_hypothesis_token_lengths()
    test_single_task_token_registration()
    test_mece_question_structure()
    test_format_investigation_state()
    print("All MECE tests passed successfully!")
