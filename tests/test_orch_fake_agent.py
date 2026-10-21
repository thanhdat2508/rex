"""Unit tests for AgentPort and FakeAgent replay (Step B0)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.orch.agent_port import AgentCommand, AgentPort, Case, CaseStatus
from rex.orch.fake_agent import FakeAgent


def test_fake_agent_implements_protocol():
    agent = FakeAgent()
    assert isinstance(agent, AgentPort)


def test_fake_agent_case_anh_a_replay():
    agent = FakeAgent.case_anh_a()
    case = Case(ticket_id="TCK-100", type_token="<FRAUD>", text="Suspicious login")

    # Round 1
    cmd1 = agent.start(case, {"geo_ip_lookup": True, "device_fingerprint": True})
    assert cmd1.kind == "CALL_TOOL"
    assert cmd1.tools == ["geo_ip_lookup", "device_fingerprint"]
    assert cmd1.s == 0.40
    assert cmd1.risk == 0.15

    # Round 2
    cmd2 = agent.observe(
        "geo_ip_lookup OK country_mismatch=1",
        {"geo_ip_lookup": {"country_mismatch": 1}},
        {"auth_history": True},
    )
    assert cmd2.kind == "CALL_TOOL"
    assert cmd2.tools == ["auth_history"]
    assert cmd2.s == 0.62
    assert cmd2.risk == 0.20

    # Round 3
    cmd3 = agent.observe(
        "auth_history OK failed_logins=5",
        {"auth_history": {"failed_logins": 5}},
        {},
    )
    assert cmd3.kind == "VERDICT"
    assert cmd3.s == 0.91
    assert cmd3.risk == 0.10
    assert cmd3.conformal_set == ["H_ato"]
    assert cmd3.verdict_probs["H_ato"] == 0.88


def test_fake_agent_exhausted_defaults_to_verdict():
    agent = FakeAgent([])
    case = Case(ticket_id="TCK-200", type_token="<FRAUD>")
    cmd = agent.start(case, {})
    assert cmd.kind == "VERDICT"
    assert cmd.s == 1.0


if __name__ == "__main__":
    test_fake_agent_implements_protocol()
    test_fake_agent_case_anh_a_replay()
    test_fake_agent_exhausted_defaults_to_verdict()
    print("PASS: test_orch_fake_agent")
