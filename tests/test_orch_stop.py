"""Unit tests and fuzz testing for Stop decision engine (Step B8)."""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.orch.agent_port import AgentCommand
from rex.orch.guard import Guard
from rex.orch.stop import Decision, StopConfig, check_epsilon_stagnation, decide


def test_auto_verdict_when_s_high():
    cfg = StopConfig()
    cmd = AgentCommand(kind="VERDICT", s=0.92, risk=0.10)
    d = decide(cmd, s_history=[0.4, 0.92], n_calls=2, round_k=2, time_left_s=10.0, high_risk=False, cfg=cfg)
    assert d.action == "AUTO_VERDICT"
    assert d.reason is None


def test_high_risk_escalation_even_if_s_high():
    cfg = StopConfig(risk_threshold=0.30)
    cmd = AgentCommand(kind="VERDICT", s=0.95, risk=0.45)
    d = decide(cmd, s_history=[0.5, 0.95], n_calls=2, round_k=2, time_left_s=10.0, high_risk=False, cfg=cfg)
    assert d.action == "ESCALATE"
    assert d.reason == "HIGH_RISK"


def test_conformal_verdict_low_s_flag_handling():
    cfg_off = StopConfig(accept_conformal_verdict=False)
    cmd = AgentCommand(kind="VERDICT", s=0.75, risk=0.10)
    d_off = decide(cmd, s_history=[0.75], n_calls=1, round_k=1, time_left_s=10.0, high_risk=False, cfg=cfg_off)
    assert d_off.action == "ESCALATE"
    assert d_off.reason == "DEADLOCK"
    assert d_off.detail == "CONFORMAL_LOW_S"

    cfg_on = StopConfig(accept_conformal_verdict=True)
    d_on = decide(cmd, s_history=[0.75], n_calls=1, round_k=1, time_left_s=10.0, high_risk=False, cfg=cfg_on)
    assert d_on.action == "AUTO_VERDICT"


def test_stall_signal_and_epsilon_stagnation():
    cfg = StopConfig(epsilon_s=0.05)

    # 1. Direct STALL signal
    cmd_stall = AgentCommand(kind="STALL", s=0.50, risk=0.10)
    d_stall = decide(cmd_stall, s_history=[0.50], n_calls=1, round_k=1, time_left_s=10.0, high_risk=False, cfg=cfg)
    assert d_stall.action == "ESCALATE"
    assert d_stall.reason == "DEADLOCK"
    assert d_stall.detail == "STALL_SIGNAL"

    # 2. Epsilon stagnation across 2 rounds
    # Round 0: 0.40, Round 1: 0.42 (delta=0.02 < 0.05), Round 2: 0.43 (delta=0.01 < 0.05)
    s_hist = [0.40, 0.42, 0.43]
    assert check_epsilon_stagnation(s_hist, epsilon=0.05) is True

    cmd_call = AgentCommand(kind="CALL_TOOL", tools=["geo_ip_lookup"], s=0.43, risk=0.10)
    d_stag = decide(cmd_call, s_history=s_hist, n_calls=2, round_k=2, time_left_s=10.0, high_risk=False, cfg=cfg)
    assert d_stag.action == "ESCALATE"
    assert d_stag.reason == "DEADLOCK"
    assert d_stag.detail == "EPSILON_STAGNATION"


def test_budget_exhausted():
    cfg = StopConfig(max_rounds=3, n_max_calls=6)
    cmd = AgentCommand(kind="CALL_TOOL", tools=["geo_ip_lookup"], s=0.60, risk=0.10)

    # Round limit reached
    d_rounds = decide(cmd, s_history=[0.60], n_calls=3, round_k=3, time_left_s=10.0, high_risk=False, cfg=cfg)
    assert d_rounds.action == "ESCALATE"
    assert d_rounds.reason == "BUDGET_EXHAUSTED"
    assert d_rounds.detail == "MAX_ROUNDS_REACHED"

    # Call limit reached
    d_calls = decide(cmd, s_history=[0.60], n_calls=6, round_k=2, time_left_s=10.0, high_risk=False, cfg=cfg)
    assert d_calls.action == "ESCALATE"
    assert d_calls.reason == "BUDGET_EXHAUSTED"
    assert d_calls.detail == "MAX_CALLS_REACHED"

    # Time limit reached
    d_time = decide(cmd, s_history=[0.60], n_calls=2, round_k=2, time_left_s=-0.01, high_risk=False, cfg=cfg)
    assert d_time.action == "ESCALATE"
    assert d_time.reason == "BUDGET_EXHAUSTED"
    assert d_time.detail == "DEADLINE_EXCEEDED"


def test_guard_filtering_in_stop_decision():
    cfg = StopConfig()
    guard = Guard(allowed_levels={"READ"})
    tools_cfg = {"tools": {"block_card": {"level": "HEAVY"}}}

    cmd = AgentCommand(kind="CALL_TOOL", tools=["block_card"], s=0.50, risk=0.10)
    d = decide(
        cmd,
        s_history=[0.50],
        n_calls=1,
        round_k=1,
        time_left_s=10.0,
        high_risk=False,
        cfg=cfg,
        guard=guard,
        tools_cfg=tools_cfg,
        available_mask={"block_card": True},
    )
    assert d.action == "ESCALATE"
    assert d.reason == "TOOL_BLOCKED"
    assert d.detail == "PROHIBITED_LEVEL"


def test_fuzz_random_agent_never_exceeds_budget():
    """Fuzz testing: random agent commands must always terminate within budget."""
    cfg = StopConfig(max_rounds=3, n_max_calls=6)
    random.seed(42)

    for _ in range(50):
        round_k = 1
        n_calls = 0
        s_hist = []
        high_risk = False
        time_left = 1.0

        while True:
            # Generate random command
            kind = random.choice(["CALL_TOOL", "CALL_TOOL", "VERDICT", "STALL"])
            s_val = random.uniform(0.1, 0.99)
            r_val = random.uniform(0.0, 0.5)
            n_tools_chosen = random.randint(1, 3)
            tools = [f"tool_{i}" for i in range(n_tools_chosen)]

            cmd = AgentCommand(kind=kind, tools=tools, s=s_val, risk=r_val)
            s_hist.append(cmd.s)

            d = decide(
                cmd,
                s_history=s_hist,
                n_calls=n_calls,
                round_k=round_k,
                time_left_s=time_left,
                high_risk=high_risk,
                cfg=cfg,
            )

            if d.action != "CONTINUE":
                # Case terminated
                assert round_k <= cfg.max_rounds + 1
                assert n_calls <= cfg.n_max_calls + 3
                break

            n_calls += len(d.tools)
            round_k += 1
            time_left -= 0.1


if __name__ == "__main__":
    test_auto_verdict_when_s_high()
    test_high_risk_escalation_even_if_s_high()
    test_conformal_verdict_low_s_flag_handling()
    test_stall_signal_and_epsilon_stagnation()
    test_budget_exhausted()
    test_guard_filtering_in_stop_decision()
    test_fuzz_random_agent_never_exceeds_budget()
    print("PASS: test_orch_stop")
