"""Data rollout generation, trajectory tracking, and PyTorch Dataset for Rex (Part 3)."""
import copy
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch
from torch.utils.data import Dataset

from .rex_format import encode_rex_input
from .rex_labels import (
    generate_redundancy_mask,
    generate_risk_label,
    generate_stall_label,
    generate_sufficiency_label,
    generate_tool_gain_labels,
)
from .rex_registry import MetricSpec, Registry


def _sample_metric_value(
    spec: MetricSpec,
    operator: Optional[str],
    threshold: Any,
    trigger: bool,
    rng: random.Random,
) -> Any:
    """Generate a metric observation value that either triggers or satisfies/avoids a rule."""
    if spec.kind == "bool":
        if operator in ("EQ", "NEQ") and threshold is not None:
            thresh_bool = bool(threshold)
            if operator == "EQ":
                return thresh_bool if trigger else (not thresh_bool)
            else:  # NEQ
                return (not thresh_bool) if trigger else thresh_bool
        return rng.choice([0, 1])

    min_v = spec.min_val if spec.min_val is not None else 0.0
    max_v = spec.max_val if spec.max_val is not None else 1.0

    if operator is None or threshold is None:
        if spec.kind == "int":
            return rng.randint(int(min_v), int(max_v))
        return round(rng.uniform(min_v, max_v), 3)

    t = float(threshold)
    eps = 0.02 if spec.kind == "float" else 1

    if operator in ("GT", "GTE"):
        if trigger:
            low = min(max_v, t + eps)
            high = max_v
            val = rng.uniform(low, high) if low < high else max_v
        else:
            low = min_v
            high = max(min_v, t - (eps if operator == "GT" else 0))
            val = rng.uniform(low, high) if low < high else min_v
    elif operator in ("LT", "LTE"):
        if trigger:
            low = min_v
            high = max(min_v, t - eps)
            val = rng.uniform(low, high) if low < high else min_v
        else:
            low = min(max_v, t + (eps if operator == "LT" else 0))
            high = max_v
            val = rng.uniform(low, high) if low < high else max_v
    elif operator == "EQ":
        if trigger:
            val = t
        else:
            alt_val = t + 2 * eps if t + 2 * eps <= max_v else t - 2 * eps
            val = alt_val
    elif operator == "NEQ":
        if trigger:
            alt_val = t + 2 * eps if t + 2 * eps <= max_v else t - 2 * eps
            val = alt_val
        else:
            val = t
    else:
        val = rng.uniform(min_v, max_v)

    if spec.kind == "int":
        return int(round(val))
    return round(float(val), 3)


def _domain_risk(domain: Any) -> float:
    """Business risk label R* for a domain: the registry value when set, else the default mapping."""
    risk = getattr(domain, "risk", None)
    return float(risk) if risk is not None else generate_risk_label(domain.type_token)


def hypothesis_risks(domain: Any) -> Dict[str, float]:
    """Registry risk of each hypothesis of a domain (domain risk when a hypothesis sets none)."""
    return {h.id: case_risk(domain, h.id) for h in domain.hypotheses}


def case_risk(domain: Any, target: str) -> float:
    """Business risk label R* of one case: the risk of acting on its true cause.

    Uses the target hypothesis' registry `risk`, falling back to the domain risk. The label then
    varies with the case, so the risk head learns E[risk | evidence] instead of a per-domain constant.
    """
    hyp = getattr(domain, "hypothesis_map", {}).get(target)
    risk = getattr(hyp, "risk", None) if hyp is not None else None
    return float(risk) if risk is not None else _domain_risk(domain)


def format_state(state: Dict[str, Any], compact: bool = True) -> str:
    """Format accumulated alert and round observations into a structured textual context."""
    if not isinstance(state, dict):
        return str(state)

    parts: List[str] = []

    # 1. Alert context
    alert = state.get("alert", {})
    if alert:
        kv = " ".join(f"{k}={v}" for k, v in sorted(alert.items()))
        parts.append(f"alert: {kv}")

    # 2. Sequential rounds
    for k in sorted(state.keys()):
        if not k.startswith("round_"):
            continue
        round_data = state[k]
        if not round_data:
            continue
        tool_parts = []
        for tool_name, metrics in sorted(round_data.items()):
            m_kv = " ".join(f"{mk}={mv}" for mk, mv in sorted(metrics.items()))
            tool_parts.append(f"{tool_name} {{{m_kv}}}")
        if tool_parts:
            parts.append(f"{k}: " + " ".join(tool_parts))

    return " | ".join(parts) if parts else "state: normal"


def sample_world(
    registry: Registry,
    domain_key: str,
    rng: random.Random,
    novel_ratio: float = 0.12,
    p_falsify: float = 0.85,
) -> Dict[str, Any]:
    """Sample an underlying incident state and simulated observations for all available tools."""
    domain = registry.get_domain(domain_key)
    known_hyps = [h for h in domain.hypotheses if not h.is_residual]

    is_novel = rng.random() < novel_ratio
    if is_novel:
        target_id = "H_novel"
    else:
        target_id = rng.choice(known_hyps).id

    # Constraints per metric: (operator, threshold, trigger)
    metric_constraints: Dict[str, Tuple[str, Any, bool]] = {}

    if is_novel:
        # Step 3.3: In OOD / H_novel incidents, all known hypotheses H_0..H_{M-1}
        # must be falsified by observations so that PoE leaves only H_novel alive.
        for h in known_hyps:
            # Pick at least one rule from h to trigger
            r = rng.choice(h.falsification_rules)
            metric_constraints[r.metric] = (r.operator, r.threshold, True)
    else:
        # In-distribution: Target H* must NOT have any of its rules triggered
        target_hyp = domain.hypothesis_map[target_id]
        for r in target_hyp.falsification_rules:
            metric_constraints[r.metric] = (r.operator, r.threshold, False)

        # For other known hypotheses, falsify them with probability p_falsify
        for h in known_hyps:
            if h.id == target_id:
                continue
            if rng.random() < p_falsify:
                # Find a rule that does not conflict with target's un-triggered metrics
                candidates = [r for r in h.falsification_rules if r.metric not in metric_constraints]
                if candidates:
                    r = rng.choice(candidates)
                    metric_constraints[r.metric] = (r.operator, r.threshold, True)
            else:
                # Keep ambiguous by ensuring none of its rules trigger
                for r in h.falsification_rules:
                    if r.metric not in metric_constraints:
                        metric_constraints[r.metric] = (r.operator, r.threshold, False)

    # Sample values for all registered metrics
    metric_values: Dict[str, Any] = {}
    for m_name, spec in registry.metrics.items():
        if m_name in metric_constraints:
            op, thresh, trigger = metric_constraints[m_name]
            metric_values[m_name] = _sample_metric_value(spec, op, thresh, trigger, rng)
        else:
            metric_values[m_name] = _sample_metric_value(spec, None, None, False, rng)

    # Structure into alert metrics and tool return outputs
    alert: Dict[str, Any] = {m: metric_values[m] for m in domain.alert_metrics if m in metric_values}
    tool_data: Dict[str, Dict[str, Any]] = {}
    for t_name, t_spec in registry.tool_specs.items():
        tool_data[t_name] = {m: metric_values[m] for m in t_spec.returns if m in metric_values}

    return {
        "domain_key": domain_key,
        "target": target_id,
        "alert": alert,
        "tool_data": tool_data,
        "is_novel": is_novel,
    }


def oracle_tools(
    registry: Registry,
    domain_key: str,
    state: Dict[str, Any],
    alive: Dict[str, bool],
    t: int,
    budgets: Tuple[int, int, int] = (3, 2, 1),
    scope_schedule: Sequence[str] = ("macro", "micro"),
    scope_fallback: bool = True,
    useful_tools: Optional[Set[str]] = None,
) -> List[str]:
    """Select next investigative tools with a greedy cover strategy.

    With `useful_tools` (the tools whose counterfactual gain* is 1) the teacher is privileged: it
    only picks tools that uncover a key fact and stops when none is left, so tool labels never
    contradict the gain labels (docs/rex-label-schema.md, labels.teacher_tools). Without it the
    teacher has no ground-truth knowledge and fills each round's budget.
    """
    domain = registry.get_domain(domain_key)
    round_idx = min(t - 1, len(budgets) - 1)
    budget = budgets[round_idx]

    # Target scope for the round
    target_scope: Optional[str] = None
    if t <= len(scope_schedule):
        target_scope = scope_schedule[t - 1]

    # Identify tools already called in prior rounds
    called_tools: Set[str] = set()
    for k, v in state.items():
        if k.startswith("round_") and isinstance(v, dict):
            called_tools.update(v.keys())

    # Get allowed tools under PoE
    tool_mask_flags = registry.tool_mask(domain_key, alive, called_tools=list(called_tools))
    candidate_tools = [tool for tool, ok in zip(registry.tools, tool_mask_flags) if ok]
    if useful_tools is not None:
        candidate_tools = [tool for tool in candidate_tools if tool in useful_tools]

    if not candidate_tools:
        return []

    # Active surviving known hypotheses
    active_hyps = [h for h in domain.hypotheses if not h.is_residual and alive.get(h.id, True)]
    if not active_hyps:
        return []

    # Utility score: number of surviving hypotheses this tool can inspect
    tool_scores: Dict[str, int] = {}
    for tool_name in candidate_tools:
        score = sum(1 for h in active_hyps if tool_name in h.associated_tools)
        tool_scores[tool_name] = score

    # Filter by scope if required
    if target_scope:
        scoped = [t for t in candidate_tools if registry.tool_specs[t].scope == target_scope and tool_scores[t] > 0]
        # Sort descending by score, tie-break by tool name
        scoped.sort(key=lambda x: (-tool_scores[x], x))
        if scoped:
            return scoped[:budget]
        elif not scope_fallback:
            return []

    # Fallback to any scope if scoped pool is empty
    fallback_candidates = [t for t in candidate_tools if tool_scores[t] > 0]
    fallback_candidates.sort(key=lambda x: (-tool_scores[x], x))
    return fallback_candidates[:budget]


def validate_sample(
    sample: Dict[str, Any],
    registry: Registry,
    budgets: Tuple[int, int, int] = (3, 2, 1),
    scope_schedule: Sequence[str] = ("macro", "micro"),
    scope_fallback: bool = True,
) -> None:
    """Strictly assert invariant guarantees on generated trajectory samples."""
    target = sample["target"]
    alive = sample["alive"]

    # 1. Target H* must never be marked eliminated
    if not alive.get(target, True):
        raise ValueError(f"Ground truth target '{target}' was marked eliminated in sample: {sample}")

    # 2. H_novel must never be marked eliminated
    if not alive.get("H_novel", True):
        raise ValueError(f"Residual slot 'H_novel' was marked eliminated in sample: {sample}")

    # 3. No label assigned to disabled tools
    allowed_set = set(sample["allowed_tools"])
    for lbl in sample["tool_labels"]:
        if lbl not in allowed_set:
            raise ValueError(f"Tool label '{lbl}' is not in allowed_tools {allowed_set}.")

    # 4. Budget constraints
    t = sample["t"]
    budget = budgets[min(t - 1, len(budgets) - 1)]
    if len(sample["tool_labels"]) > budget:
        raise ValueError(
            f"Number of tool labels ({len(sample['tool_labels'])}) exceeds round budget ({budget})."
        )

    # 5. Scope adherence (when not falling back)
    if not scope_fallback and t <= len(scope_schedule):
        expected_scope = scope_schedule[t - 1]
        for lbl in sample["tool_labels"]:
            lbl_scope = registry.tool_specs[lbl].scope
            if lbl_scope != expected_scope:
                raise ValueError(
                    f"Tool '{lbl}' has scope '{lbl_scope}', expected '{expected_scope}' in round {t}."
                )


def rollout(
    registry: Registry,
    domain_key: str,
    rng: random.Random,
    case_id: str,
    budgets: Tuple[int, int, int] = (3, 2, 1),
    scope_schedule: Sequence[str] = ("macro", "micro"),
    novel_ratio: float = 0.12,
    p_falsify: float = 0.85,
    scope_fallback: bool = True,
    privileged_teacher: bool = True,
) -> List[Dict[str, Any]]:
    """Simulate a complete 1-3 turn investigation incident rollout.

    `privileged_teacher` (default) labels only the tools with gain* = 1 (see oracle_tools), while the
    behaviour policy that produces the observations keeps the exploratory budget-filling choice, so
    training states include rounds that uncover nothing, as at inference. `called_tools` records
    what the behaviour policy called; `tool_labels` is what the teacher recommends.
    """
    world = sample_world(registry, domain_key, rng, novel_ratio=novel_ratio, p_falsify=p_falsify)
    state: Dict[str, Any] = {"alert": world["alert"]}

    # Collect all ground-truth metric values from world rollout for counterfactual labeling
    all_observations: Dict[str, Any] = dict(world["alert"])
    for t_data in world["tool_data"].values():
        all_observations.update(t_data)

    domain = registry.get_domain(domain_key)
    falsification_rules = {h.id: h.falsification_rules for h in domain.hypotheses}
    tool_specs_list = [registry.tool_specs[t_name] for t_name in registry.tools]
    tool_hyp_map: Dict[str, List[str]] = {t_name: [] for t_name in registry.tools}
    for h in domain.hypotheses:
        for t in h.associated_tools:
            if t in tool_hyp_map:
                tool_hyp_map[t].append(h.id)

    # Initial Popperian evaluation on alert metrics
    alive, eliminated_by = registry.falsify(domain_key, world["alert"])

    samples: List[Dict[str, Any]] = []
    called_tools: List[str] = []

    terminal_recorded = False
    for t in (1, 2, 3):
        allowed = [
            tool
            for tool, ok in zip(registry.tools, registry.tool_mask(domain_key, alive, called_tools))
            if ok
        ]

        alive_list = [k for k, v in alive.items() if v]
        s_star = generate_sufficiency_label(alive_list, world["target"])
        r_star = case_risk(domain, world["target"])
        gain_dict = generate_tool_gain_labels(
            alive_list,
            tool_specs_list,
            all_observations,
            falsification_rules,
            mask_unobserved=True,
        )

        oracle_kwargs = dict(
            registry=registry,
            domain_key=domain_key,
            state=state,
            alive=alive,
            t=t,
            budgets=budgets,
            scope_schedule=scope_schedule,
            scope_fallback=scope_fallback,
        )
        # Behaviour policy: what is actually called, so the next states look like inference (some
        # rounds uncover nothing). Teacher labels: with privileged_teacher, only the useful tools.
        behaviour = oracle_tools(**oracle_kwargs)
        if privileged_teacher:
            labels = oracle_tools(**oracle_kwargs, useful_tools={tool for tool, g in gain_dict.items() if g > 0.0})
            budget = budgets[min(t - 1, len(budgets) - 1)]
            calls = (list(labels) + [tool for tool in behaviour if tool not in labels])[:budget]
        else:
            labels = behaviour
            calls = list(behaviour)
        stall_star = generate_stall_label(s_star, gain_dict, allowed_tools=allowed)
        redundancy_mask = generate_redundancy_mask(registry.tools, called_tools, alive_list, tool_hyp_map)
        hyp_true = {h_id: (1.0 if alive.get(h_id, True) else 0.0) for h_id in alive}

        step_labels = {
            "s_star": float(s_star),
            "r_star": float(r_star),
            "gain_star": {tool: gain_dict.get(tool, -100.0) for tool in registry.tools},
            "stall_star": float(stall_star),
            "redundancy_mask": redundancy_mask,
            "hyp_true": hyp_true,
            "hyp_risk": hypothesis_risks(domain),
        }

        if not calls:
            # Step 3.1: When oracle returns empty at round t, terminal sample carries t (not t + 1)
            terminal_sample = {
                "case_id": case_id,
                "t": t,
                "type_token": domain_key,
                "state": copy.deepcopy(state),
                "alive": dict(alive),
                "allowed_tools": allowed,
                "tool_labels": [],  # Stop calling tools
                "label_scope": "terminal",
                "observations": {},
                "target": world["target"],
                "eliminated_by": copy.deepcopy(eliminated_by),
                "labels": copy.deepcopy(step_labels),
            }
            validate_sample(
                terminal_sample,
                registry,
                budgets=budgets,
                scope_schedule=scope_schedule,
                scope_fallback=scope_fallback,
            )
            samples.append(terminal_sample)
            terminal_recorded = True
            break

        # Observations for this turn
        obs = {tool: world["tool_data"][tool] for tool in calls}

        # Determine label scope ("none": the behaviour policy explores but no tool is worth calling)
        scopes = {registry.tool_specs[t_name].scope for t_name in labels}
        label_scope = (scopes.pop() if len(scopes) == 1 else "mixed") if scopes else "none"

        step_sample = {
            "case_id": case_id,
            "t": t,
            "type_token": domain_key,
            "state": copy.deepcopy(state),
            "alive": dict(alive),
            "allowed_tools": allowed,
            "tool_labels": list(labels),
            "called_tools": list(calls),
            "label_scope": label_scope,
            "observations": copy.deepcopy(obs),
            "target": world["target"],
            "eliminated_by": copy.deepcopy(eliminated_by),
            "labels": copy.deepcopy(step_labels),
        }
        validate_sample(
            step_sample,
            registry,
            budgets=budgets,
            scope_schedule=scope_schedule,
            scope_fallback=scope_fallback,
        )
        samples.append(step_sample)

        # Update state and PoE mask for next round
        state[f"round_{t}"] = obs
        called_tools.extend(calls)
        alive, eliminated_by = registry.falsify(domain_key, state, current_alive=alive)

    if not terminal_recorded:
        # All 3 rounds called tools; record final terminal sample at round 3
        terminal_allowed = [
            tool
            for tool, ok in zip(registry.tools, registry.tool_mask(domain_key, alive, called_tools))
            if ok
        ]
        alive_list = [k for k, v in alive.items() if v]
        s_star = generate_sufficiency_label(alive_list, world["target"])
        r_star = case_risk(domain, world["target"])
        gain_dict = generate_tool_gain_labels(
            alive_list,
            tool_specs_list,
            all_observations,
            falsification_rules,
            mask_unobserved=True,
        )
        stall_star = generate_stall_label(s_star, gain_dict, allowed_tools=terminal_allowed)
        redundancy_mask = generate_redundancy_mask(registry.tools, called_tools, alive_list, tool_hyp_map)
        hyp_true = {h_id: (1.0 if alive.get(h_id, True) else 0.0) for h_id in alive}

        term_labels = {
            "s_star": float(s_star),
            "r_star": float(r_star),
            "gain_star": {tool: gain_dict.get(tool, -100.0) for tool in registry.tools},
            "stall_star": float(stall_star),
            "redundancy_mask": redundancy_mask,
            "hyp_true": hyp_true,
            "hyp_risk": hypothesis_risks(domain),
        }
        terminal_sample = {
            "case_id": case_id,
            "t": 3,
            "type_token": domain_key,
            "state": copy.deepcopy(state),
            "alive": dict(alive),
            "allowed_tools": terminal_allowed,
            "tool_labels": [],  # Stop calling tools
            "label_scope": "terminal",
            "observations": {},
            "target": world["target"],
            "eliminated_by": copy.deepcopy(eliminated_by),
            "labels": term_labels,
        }
        validate_sample(
            terminal_sample,
            registry,
            budgets=budgets,
            scope_schedule=scope_schedule,
            scope_fallback=scope_fallback,
        )
        samples.append(terminal_sample)

    return samples


def generate(
    registry: Registry,
    n_cases: int,
    rng_or_seed: Union[random.Random, int] = 42,
    budgets: Tuple[int, int, int] = (3, 2, 1),
    scope_schedule: Sequence[str] = ("macro", "micro"),
    novel_ratio: float = 0.12,
    p_falsify: float = 0.85,
    scope_fallback: bool = True,
    privileged_teacher: bool = True,
) -> List[Dict[str, Any]]:
    """Generate multi-turn trajectory samples across all registered domains."""
    rng = rng_or_seed if isinstance(rng_or_seed, random.Random) else random.Random(rng_or_seed)
    domain_keys = sorted(registry.domains.keys())

    all_samples: List[Dict[str, Any]] = []
    for case_idx in range(n_cases):
        domain_key = domain_keys[case_idx % len(domain_keys)]
        case_id = f"case-{case_idx + 1:06d}"
        case_samples = rollout(
            registry=registry,
            domain_key=domain_key,
            rng=rng,
            case_id=case_id,
            budgets=budgets,
            scope_schedule=scope_schedule,
            novel_ratio=novel_ratio,
            p_falsify=p_falsify,
            scope_fallback=scope_fallback,
            privileged_teacher=privileged_teacher,
        )
        all_samples.extend(case_samples)

    return all_samples


def split_by_case(
    samples: List[Dict[str, Any]],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split trajectory dataset into train and validation sets strictly by case_id to prevent leakage."""
    case_ids = sorted(list({s["case_id"] for s in samples}))
    rng = random.Random(seed)
    rng.shuffle(case_ids)

    n_val = max(1, int(len(case_ids) * val_ratio))
    val_cases = set(case_ids[:n_val])

    train_samples = [s for s in samples if s["case_id"] not in val_cases]
    val_samples = [s for s in samples if s["case_id"] in val_cases]

    return train_samples, val_samples


def stats(samples: List[Dict[str, Any]], registry: Registry) -> Dict[str, Any]:
    """Calculate comprehensive dataset statistics and metrics."""
    cases = set(s["case_id"] for s in samples)
    n_cases = len(cases)
    n_samples = len(samples)

    novel_cases = set(s["case_id"] for s in samples if s["target"] == "H_novel")
    novel_ratio = len(novel_cases) / max(1, n_cases)

    # Tool counts per round
    round_tools: Dict[int, List[int]] = {1: [], 2: [], 3: []}
    scope_counts: Dict[int, Dict[str, int]] = {1: {}, 2: {}, 3: {}}

    for s in samples:
        t = s["t"]
        if t in round_tools:
            labels = s["tool_labels"]
            round_tools[t].append(len(labels))
            for lbl in labels:
                sc = registry.tool_specs[lbl].scope
                scope_counts[t][sc] = scope_counts[t].get(sc, 0) + 1

    avg_tools_round = {
        r: (sum(counts) / len(counts) if counts else 0.0)
        for r, counts in round_tools.items()
    }

    # Verify OOD PoE elimination: in novel cases, how many terminal samples have all known hypotheses eliminated?
    novel_terminal_samples = [
        s for s in samples
        if s["case_id"] in novel_cases and s["tool_labels"] == []
    ]
    novel_all_known_eliminated = 0
    for s in novel_terminal_samples:
        d = registry.get_domain(s["type_token"])
        known_ids = [h.id for h in d.hypotheses if not h.is_residual]
        if all(not s["alive"].get(h_id, True) for h_id in known_ids):
            novel_all_known_eliminated += 1

    elim_rate = novel_all_known_eliminated / max(1, len(novel_terminal_samples))

    return {
        "n_cases": n_cases,
        "n_samples": n_samples,
        "novel_ratio": round(novel_ratio, 4),
        "avg_tools_per_round": {f"round_{r}": round(v, 2) for r, v in avg_tools_round.items()},
        "scope_distribution": scope_counts,
        "novel_known_eliminated_rate": round(elim_rate, 4),
    }


def write_jsonl(path: Union[str, Path], samples: Sequence[Dict[str, Any]]) -> None:
    """Save sample records to a JSON Lines file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load sample records from a JSON Lines file."""
    p = Path(path)
    records: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class RexTrajectoryDataset(Dataset):
    """PyTorch Dataset wrapping Rex multi-turn trajectory records with dynamic hypothesis permutation."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tok: Any,
        tools: Sequence[str],
        registry: Optional[Registry] = None,
        max_len: int = 512,
        head_max_len: int = 192,
        shuffle_hypotheses: bool = True,
        truncate_left: bool = True,
        base_seed: int = 42,
    ):
        self.samples = samples
        self.tok = tok
        self.tools = list(tools)
        self.registry = registry
        self.max_len = max_len
        self.head_max_len = head_max_len
        self.shuffle_hypotheses = shuffle_hypotheses
        self.truncate_left = truncate_left
        self.base_seed = base_seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Update training epoch to generate distinct random hypothesis permutations."""
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        domain_key = sample["type_token"]

        if self.registry is not None:
            domain = self.registry.get_domain(domain_key)
            instruction = domain.instruction
            hypotheses = domain.hypotheses_dict
        else:
            # Reconstruct from sample fields
            instruction = f"diagnose {domain_key}"
            hypotheses = {k: f"hypothesis {k}" for k in sample["alive"].keys()}
            if "H_novel" not in hypotheses:
                hypotheses["H_novel"] = "novel anomaly"

        hyp_keys = list(hypotheses.keys())
        target_name = sample["target"]

        # Option ordering: if shuffle_hypotheses, permute hypotheses order
        if self.shuffle_hypotheses:
            rng = random.Random(self.base_seed + self.epoch * 1000003 + idx)
            perm = list(range(len(hyp_keys)))
            rng.shuffle(perm)
            option_order = perm
            ordered_keys = [hyp_keys[i] for i in perm]
            target_idx = ordered_keys.index(target_name)
        else:
            option_order = None
            target_idx = hyp_keys.index(target_name)

        # Tool mask
        tool_mask = None
        if "allowed_tools" in sample:
            allowed_set = set(sample["allowed_tools"])
            tool_mask = [t in allowed_set for t in self.tools]

        # Format state
        state_str = format_state(sample["state"])

        encoded = encode_rex_input(
            tok=self.tok,
            type_token=domain_key,
            instruction=instruction,
            hypotheses=hypotheses,
            state=state_str,
            alive=sample["alive"],
            max_len=self.max_len,
            head_max_len=self.head_max_len,
            truncate_left=self.truncate_left,
            option_order=option_order,
            tool_mask=tool_mask,
        )

        ordered_keys_for_hyp = ordered_keys if self.shuffle_hypotheses else hyp_keys
        item = build_step_labels(sample, self.tools, ordered_keys_for_hyp, option_order=option_order)
        item.update(
            input_ids=encoded["input_ids"],
            markers=encoded["markers"],
            hyp_alive=encoded["hyp_alive"],
            tool_mask=encoded["tool_mask"],
            target_verdict=target_idx,
        )
        return item


def build_step_labels(
    sample: Dict[str, Any],
    tools: Sequence[str],
    ordered_keys: Sequence[str],
    option_order: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Supervision vectors of one trajectory step, with hypothesis-indexed fields in `ordered_keys` order.

    Shared by RexTrajectoryDataset (one step per item) and the trajectory trainer (one case per item,
    hypothesis order fixed by the case's round-1 layout).
    """
    # Build multi-hot target vector for tools [N_tools] (float32)
    target_tools_set = set(sample.get("tool_labels", []))
    tool_labels_vec = [1.0 if t in target_tools_set else 0.0 for t in tools]
    is_terminal = sample.get("label_scope") == "terminal" if "label_scope" in sample else not target_tools_set

    # Counterfactual supervision labels (T12, T3)
    labels_info = sample.get("labels", {})
    s_val = float(labels_info.get("s_star", sample.get("s_star", 0.0)))
    r_val = float(labels_info.get("r_star", sample.get("r_star", 0.1)))

    raw_gain = labels_info.get("gain_star", sample.get("gain_star", {}))
    if isinstance(raw_gain, dict):
        gain_vec = [float(raw_gain.get(t, -100.0)) for t in tools]
    elif isinstance(raw_gain, list):
        gain_vec = [float(x) for x in raw_gain]
    else:
        gain_vec = [-100.0] * len(tools)

    red_mask = labels_info.get("redundancy_mask", sample.get("redundancy_mask", [False] * len(tools)))

    raw_hyp = labels_info.get("hyp_true", sample.get("hyp_true", {}))
    if isinstance(raw_hyp, dict):
        hyp_true_vec = [float(raw_hyp.get(k, 1.0)) for k in ordered_keys]
    elif isinstance(raw_hyp, list) and option_order is not None:
        hyp_true_vec = [float(raw_hyp[i]) for i in option_order if i < len(raw_hyp)]
    else:
        hyp_true_vec = [1.0 if sample["alive"].get(k, True) else 0.0 for k in ordered_keys]

    tool_mask = None
    if "allowed_tools" in sample:
        allowed_set = set(sample["allowed_tools"])
        tool_mask = [t in allowed_set for t in tools]

    raw_hyp_risk = labels_info.get("hyp_risk", {})
    hyp_risk_vec = [float(raw_hyp_risk.get(k, -100.0)) for k in ordered_keys]

    return {
        "hyp_risk": hyp_risk_vec,
        "stall_label": float(labels_info.get("stall_star", sample.get("stall_star", 0.0))),
        "tool_labels": tool_labels_vec,
        "target_verdict": list(ordered_keys).index(sample["target"]),
        "round": sample["t"],
        "is_terminal": is_terminal,
        "case_id": sample["case_id"],
        "s_label": s_val,
        "r_label": r_val,
        "gain_labels": gain_vec,
        "redundancy_mask": red_mask,
        "hyp_true": hyp_true_vec,
        "hyp_alive": [bool(sample["alive"].get(k, True)) for k in ordered_keys],
        "tool_mask": tool_mask,
    }


def collate_rex_train(
    batch: Sequence[Dict[str, Any]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """Collate trajectory dataset items into padded batch tensors for Rex multi-task training.

    Returns:
    - input_ids: [B, max_len]
    - attention_mask: [B, max_len]
    - marker_pos: [B, max_markers]
    - mask_token_indices: alias for marker_pos (Step 3.4 contract)
    - marker_mask: [B, max_markers] bool
    - hyp_alive: [B, max_markers] bool
    - poe_mask: [B, max_markers] long binary mask (0 or 1, for PoEMaskedCrossEntropyLoss)
    - tool_labels: [B, n_tools] float32 multi-hot target
    - target_verdict: [B] long scalar target index
    - qtype: [B] long
    - round: [B] long
    - is_terminal: [B] bool flag (True for terminal verdict decisions, False for intermediate tool dispatch turns)
    - tool_mask: [B, n_tools] bool (optional)
    - s_labels: [B, 1] float32
    - r_labels: [B, 1] float32
    - gain_labels: [B, n_tools] float32
    - redundancy_masks: [B, n_tools] bool
    - hyp_labels: [B, max_markers] float32
    """
    b = len(batch)
    max_len = max(len(item["input_ids"]) for item in batch)
    max_markers = max(len(item["markers"]) for item in batch)
    n_tools = len(batch[0]["tool_labels"])

    input_ids = torch.full((b, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((b, max_len), dtype=torch.long)
    marker_pos = torch.zeros((b, max_markers), dtype=torch.long)
    marker_mask = torch.zeros((b, max_markers), dtype=torch.bool)
    hyp_alive = torch.zeros((b, max_markers), dtype=torch.bool)
    tool_labels = torch.zeros((b, n_tools), dtype=torch.float32)
    target_verdict = torch.zeros(b, dtype=torch.long)
    rounds = torch.zeros(b, dtype=torch.long)
    is_terminal = torch.tensor([bool(item.get("is_terminal", False)) for item in batch], dtype=torch.bool)

    s_labels = torch.zeros((b, 1), dtype=torch.float32)
    r_labels = torch.zeros((b, 1), dtype=torch.float32)
    gain_labels = torch.zeros((b, n_tools), dtype=torch.float32)
    redundancy_masks = torch.zeros((b, n_tools), dtype=torch.bool)
    hyp_labels = torch.zeros((b, max_markers), dtype=torch.float32)

    has_tool_mask = any(item.get("tool_mask") is not None for item in batch)
    tool_mask_tensor = torch.ones((b, n_tools), dtype=torch.bool) if has_tool_mask else None

    for i, item in enumerate(batch):
        ids = item["input_ids"]
        seq_len = len(ids)
        input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :seq_len] = 1

        markers = item["markers"]
        m_len = len(markers)
        marker_pos[i, :m_len] = torch.tensor(markers, dtype=torch.long)
        marker_mask[i, :m_len] = True

        alive_flags = item.get("hyp_alive", [True] * m_len)
        hyp_alive[i, :m_len] = torch.tensor(alive_flags, dtype=torch.bool)

        tool_labels[i] = torch.tensor(item["tool_labels"], dtype=torch.float32)
        target_verdict[i] = item["target_verdict"]
        rounds[i] = item.get("round", 1)

        s_labels[i, 0] = float(item.get("s_label", 0.0))
        r_labels[i, 0] = float(item.get("r_label", 0.1))

        g_l = item.get("gain_labels", [-100.0] * n_tools)
        gain_labels[i, : len(g_l)] = torch.tensor(g_l, dtype=torch.float32)

        r_m = item.get("redundancy_mask", [False] * n_tools)
        redundancy_masks[i, : len(r_m)] = torch.tensor(r_m, dtype=torch.bool)

        h_t = item.get("hyp_true", [1.0] * m_len)
        hyp_labels[i, : len(h_t)] = torch.tensor(h_t, dtype=torch.float32)

        if tool_mask_tensor is not None and item.get("tool_mask") is not None:
            tm = item["tool_mask"]
            tool_mask_tensor[i, : len(tm)] = torch.tensor(tm, dtype=torch.bool)

    poe_mask = hyp_alive.long()

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "mask_token_indices": marker_pos,  # Step 3.4 spec alias
        "marker_mask": marker_mask,
        "hyp_alive": hyp_alive,
        "poe_mask": poe_mask,  # Step 3.4 spec binary tensor
        "tool_labels": tool_labels,
        "target_verdict": target_verdict,
        "qtype": torch.zeros(b, dtype=torch.long),
        "round": rounds,
        "is_terminal": is_terminal,
        "s_labels": s_labels,
        "r_labels": r_labels,
        "gain_labels": gain_labels,
        "redundancy_masks": redundancy_masks,
        "hyp_labels": hyp_labels,
    }
    if tool_mask_tensor is not None:
        result["tool_mask"] = tool_mask_tensor

    return result
