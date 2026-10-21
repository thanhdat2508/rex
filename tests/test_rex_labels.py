"""Unit tests for Rex counterfactual label generation functions (Step A9)."""
from rex.rex_labels import (
    generate_redundancy_mask,
    generate_stall_label,
    generate_sufficiency_label,
    generate_tool_gain_labels,
)
from rex.rex_registry import FalsificationRule, ToolSpec


def test_generate_sufficiency_label():
    """Verify sufficiency label computation for standard, novel, and ambiguous states."""
    # 1. Singleton winner matching target
    assert generate_sufficiency_label(["H_1"], target_hypothesis="H_1") == 1.0

    # 2. Target already eliminated
    assert generate_sufficiency_label(["H_2", "H_3"], target_hypothesis="H_1") == 0.0

    # 3. Multiple survivors (ambiguous)
    assert generate_sufficiency_label(["H_1", "H_2"], target_hypothesis="H_1") == 0.0

    # 4. Target falsified flag
    assert generate_sufficiency_label(["H_1"], target_hypothesis="H_1", target_falsified=True) == 0.0

    # 5. OOD / Novel case: all cataloged hypotheses refuted, target is H_novel
    assert generate_sufficiency_label(["H_novel"], target_hypothesis="H_novel") == 1.0
    assert generate_sufficiency_label(["H_1", "H_novel"], target_hypothesis="H_novel") == 0.0


def test_generate_tool_gain_labels():
    """Verify counterfactual epistemic gain calculation across tools."""
    tool_specs = [
        ToolSpec(name="t_macro", returns=["err_rate", "vol"], scope="macro"),
        ToolSpec(name="t_micro", returns=["db_latency"], scope="micro"),
        ToolSpec(name="t_idle", returns=["unused_metric"], scope="macro"),
    ]

    rules = {
        "H_1": [FalsificationRule(metric="err_rate", operator="GT", threshold=5.0)],
        "H_2": [FalsificationRule(metric="vol", operator="LT", threshold=100.0)],
        "H_3": [FalsificationRule(metric="db_latency", operator="GT", threshold=50.0)],
    }

    # World rollout observations:
    # err_rate = 10.0 (> 5.0 -> H_1 eliminated)
    # vol = 50.0 (< 100.0 -> H_2 eliminated)
    # db_latency = 10.0 (not > 50.0 -> H_3 survives)
    observations = {
        "err_rate": 10.0,
        "vol": 50.0,
        "db_latency": 10.0,
    }

    alive = ["H_1", "H_2", "H_3", "H_novel"]  # total 4 alive

    gains = generate_tool_gain_labels(alive, tool_specs, observations, rules)

    # t_macro returns err_rate and vol -> eliminates H_1 and H_2 (2 out of 4) -> 2/4 = 0.5
    assert gains["t_macro"] == 0.5

    # t_micro returns db_latency -> eliminates 0 out of 4 -> 0.0
    assert gains["t_micro"] == 0.0

    # t_idle has no relevant observations -> 0.0
    assert gains["t_idle"] == 0.0


def test_generate_stall_label():
    """Verify ground truth STALL signal generation."""
    # When sufficiency is achieved, STALL is always 0.0
    assert generate_stall_label(s_star=1.0, tool_gains={"t1": 0.0}) == 0.0

    # When S* < threshold and all tools yield zero gain -> STALL = 1.0
    tool_gains_stuck = {"t1": 0.0, "t2": 0.0}
    assert generate_stall_label(s_star=0.0, tool_gains=tool_gains_stuck) == 1.0

    # When S* < threshold but tool t1 yields significant gain above cost -> STALL = 0.0
    tool_gains_active = {"t1": 0.4, "t2": 0.0}
    tool_costs = {"t1": 0.1, "t2": 0.0}
    assert generate_stall_label(s_star=0.0, tool_gains=tool_gains_active, tool_costs=tool_costs) == 0.0


def test_generate_redundancy_mask():
    """Verify redundancy mask generation for called and refuted tool branches."""
    all_tools = ["t_disk", "t_net", "t_auth", "t_general"]
    called_tools = ["t_disk"]
    alive_hypotheses = ["H_auth"]

    tool_hyp_mapping = {
        "t_disk": ["H_disk"],
        "t_net": ["H_net"],
        "t_auth": ["H_auth"],
        "t_general": [],
    }

    mask = generate_redundancy_mask(
        all_tools=all_tools,
        called_tools=called_tools,
        alive_hypotheses=alive_hypotheses,
        tool_hyp_mapping=tool_hyp_mapping,
    )

    # t_disk was called -> True
    # t_net's hypothesis H_net is eliminated -> True
    # t_auth's hypothesis H_auth is alive -> False
    # t_general has no specific hypothesis and is uncalled -> False
    assert mask == [True, True, False, False]
