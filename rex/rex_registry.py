"""Popperian hypothesis registry, metric definitions, and falsification engine for Rex."""
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

try:
    import yaml
except ImportError:
    yaml = None  # Optional dependency, JSON is primary


ALLOWED_OPERATORS: Set[str] = {"GT", "LT", "GTE", "LTE", "EQ", "NEQ", "IN", "NOT_IN"}


@dataclass(frozen=True)
class FalsificationRule:
    """Popperian falsification rule defining an elimination condition for a hypothesis."""

    metric: str
    operator: str
    threshold: Any

    def __post_init__(self):
        if self.operator not in ALLOWED_OPERATORS:
            raise ValueError(
                f"Unknown falsification operator '{self.operator}'. "
                f"Allowed operators: {sorted(ALLOWED_OPERATORS)}"
            )

    def eval(self, value: Any) -> bool:
        """Evaluate observation value against rule condition.

        Returns True if the condition is MET, meaning the hypothesis is FALSIFIED / ELIMINATED.
        """
        if value is None:
            return False

        op = self.operator
        thresh = self.threshold

        if op == "GT":
            return float(value) > float(thresh)
        elif op == "LT":
            return float(value) < float(thresh)
        elif op == "GTE":
            return float(value) >= float(thresh)
        elif op == "LTE":
            return float(value) <= float(thresh)
        elif op == "EQ":
            if isinstance(thresh, bool) or isinstance(value, bool):
                return bool(value) is bool(thresh)
            return value == thresh
        elif op == "NEQ":
            if isinstance(thresh, bool) or isinstance(value, bool):
                return bool(value) is not bool(thresh)
            return value != thresh
        elif op == "IN":
            return value in thresh
        elif op == "NOT_IN":
            return value not in thresh
        return False


@dataclass(frozen=True)
class MetricSpec:
    """Specification of a measurable metric in the environment."""

    name: str
    kind: str  # "float", "int", "bool"
    min_val: Optional[float] = None
    max_val: Optional[float] = None


@dataclass(frozen=True)
class ToolSpec:
    """Specification of an investigative tool's returned metrics and operational scope."""

    name: str
    returns: List[str]
    scope: str = "macro"  # "macro" or "micro"
    general: bool = False
    description: str = ""
    level: str = "READ"   # "READ", "LIGHT", or "HEAVY"
    cost: float = 0.0


@dataclass(frozen=True)
class HypothesisSpec:
    """Hypothesis definition including associated tools and falsification criteria."""

    id: str
    text: str
    associated_tools: List[str] = field(default_factory=list)
    falsification_rules: List[FalsificationRule] = field(default_factory=list)
    is_residual: bool = False
    risk: Optional[float] = None  # business risk R* when this is the true cause; None -> domain risk


@dataclass(frozen=True)
class DomainSpec:
    """Investigation domain schema corresponding to a type token (e.g. <TYPE_PROMO_ABUSE>)."""

    type_token: str
    instruction: str
    alert_metrics: List[str]
    hypotheses: List[HypothesisSpec]
    risk: Optional[float] = None  # business risk label R* in [0, 1]; None falls back to the default mapping

    @property
    def hypotheses_dict(self) -> Dict[str, str]:
        """Mapping from hypothesis ID to descriptive text."""
        return {h.id: h.text for h in self.hypotheses}

    @property
    def hypothesis_map(self) -> Dict[str, HypothesisSpec]:
        """Mapping from hypothesis ID to full HypothesisSpec."""
        return {h.id: h for h in self.hypotheses}


def _flatten_observations(observations: Any) -> Dict[str, Any]:
    """Recursively flatten nested observation dictionaries into a single metric->value mapping."""
    flat: Dict[str, Any] = {}
    if not isinstance(observations, dict):
        return flat

    for k, v in observations.items():
        if isinstance(v, dict):
            flat.update(_flatten_observations(v))
        else:
            flat[k] = v
    return flat


class Registry:
    """Central registry maintaining tools, metrics, domains, and Popperian elimination logic."""

    def __init__(
        self,
        tools: List[str],
        type_tokens: Dict[str, str],
        metrics: Dict[str, MetricSpec],
        tool_specs: Dict[str, ToolSpec],
        scope_schedule: List[str],
        domains: Dict[str, DomainSpec],
        tool_hidden: Optional[int] = None,
        tool_dropout: float = 0.1,
        tool_prior: Optional[float] = None,
    ):
        self.tools = list(tools)
        self.tool_to_idx = {name: i for i, name in enumerate(self.tools)}
        self.type_tokens = dict(type_tokens)
        self.metrics = dict(metrics)
        self.tool_specs = dict(tool_specs)
        self.scope_schedule = list(scope_schedule)
        self.domains = dict(domains)
        self.tool_hidden = tool_hidden
        self.tool_dropout = tool_dropout
        self.tool_prior = tool_prior

    def check_model_tools(self, model_tools: Sequence[str]) -> None:
        """Validate that the registry tools match the model tool head output order and vocabulary."""
        reg_tools = list(self.tools)
        mod_tools = list(model_tools)
        if reg_tools != mod_tools:
            if set(reg_tools) != set(mod_tools):
                diff_reg = set(reg_tools) - set(mod_tools)
                diff_mod = set(mod_tools) - set(reg_tools)
                raise ValueError(
                    f"Registry tools and model tools do not match! "
                    f"In registry only: {sorted(diff_reg)}, in model only: {sorted(diff_mod)}"
                )
            # Permutation mismatch warning
            raise ValueError(
                f"Registry tools and model tools have differing order.\n"
                f"Registry: {reg_tools}\nModel: {mod_tools}"
            )

    def get_domain(self, type_token: str) -> DomainSpec:
        """Retrieve domain specification for a given domain event type token."""
        if type_token not in self.domains:
            raise KeyError(
                f"Domain '{type_token}' not configured in registry. "
                f"Available domains: {list(self.domains.keys())}"
            )
        return self.domains[type_token]

    def falsify(
        self,
        domain_key: str,
        observations: Dict[str, Any],
        current_alive: Optional[Dict[str, bool]] = None,
    ) -> Tuple[Dict[str, bool], Dict[str, List[str]]]:
        """Evaluate observations against falsification rules to eliminate hypotheses.

        Parameters
        ----------
        domain_key : str
            Domain type token (e.g. "<TYPE_PROMO_ABUSE>").
        observations : Dict[str, Any]
            Accumulated evidence observations (flat or nested tool dictionaries).
        current_alive : Optional[Dict[str, bool]]
            Current hypothesis survival mask. If None, all hypotheses start as alive.

        Returns
        -------
        Tuple[Dict[str, bool], Dict[str, List[str]]]
            Updated alive dictionary and eliminated_by proof trace dictionary.
        """
        domain = self.get_domain(domain_key)
        flat_obs = _flatten_observations(observations)

        alive: Dict[str, bool] = {}
        eliminated_by: Dict[str, List[str]] = {}

        # Initialize survival status
        for h in domain.hypotheses:
            if current_alive is not None and h.id in current_alive:
                alive[h.id] = bool(current_alive[h.id])
            else:
                alive[h.id] = True

        # Process each hypothesis
        for h in domain.hypotheses:
            # H_novel (residual) cannot ever be eliminated
            if h.is_residual or h.id == "H_novel":
                alive[h.id] = True
                continue

            # If already eliminated in a prior turn, keep eliminated
            if not alive[h.id]:
                continue

            # Check falsification conditions
            for rule in h.falsification_rules:
                if rule.metric in flat_obs:
                    val = flat_obs[rule.metric]
                    if rule.eval(val):
                        alive[h.id] = False
                        trace = f"{rule.metric} {rule.operator} {rule.threshold} (observed {val})"
                        eliminated_by.setdefault(h.id, []).append(trace)
                        break

        # Invariant: H_novel must always remain alive
        alive["H_novel"] = True

        return alive, eliminated_by

    def tool_mask(
        self,
        domain_key: str,
        alive: Dict[str, bool],
        called_tools: Optional[Sequence[str]] = None,
    ) -> List[bool]:
        """Compute boolean availability mask for tools given the surviving hypotheses.

        A tool is available (True) if:
        1. It has NOT been called already.
        2. It is designated as a general tool OR is associated with at least one currently alive hypothesis.

        Returns
        -------
        List[bool]
            Boolean mask of length len(self.tools).
        """
        domain = self.get_domain(domain_key)
        called_set = set(called_tools or [])

        # Active tools needed by surviving non-residual hypotheses
        active_tools: Set[str] = set()
        for h in domain.hypotheses:
            if not h.is_residual and alive.get(h.id, True):
                active_tools.update(h.associated_tools)

        mask: List[bool] = []
        for t in self.tools:
            if t in called_set:
                mask.append(False)
            elif self.tool_specs.get(t) and self.tool_specs[t].general:
                mask.append(True)
            elif t in active_tools:
                mask.append(True)
            else:
                mask.append(False)

        return mask


def load_registry(path_or_dict: Union[str, Path, dict]) -> Registry:
    """Load and strictly validate a Rex registry from file path or dictionary.

    Performs comprehensive schema validation:
    - Unique non-empty tools list.
    - All tool specs reference valid tools and return registered metrics.
    - Domains end with residual 'H_novel'.
    - Known hypotheses possess at least one falsification rule.
    - Falsification metrics are observable (either in alert_metrics or tool returns).
    """
    if isinstance(path_or_dict, (str, Path)):
        p = Path(path_or_dict)
        if not p.is_file():
            raise FileNotFoundError(f"Registry file not found: {p}")
        text = p.read_text(encoding="utf-8")
        if p.suffix in (".yaml", ".yml"):
            if yaml is None:
                raise ImportError("PyYAML is required to parse YAML registry files.")
            raw = yaml.safe_load(text)
        else:
            raw = json.loads(text)
    elif isinstance(path_or_dict, dict):
        raw = dict(path_or_dict)
    else:
        raise TypeError(f"Expected path or dict, got {type(path_or_dict)}")

    # 1. Validate tools
    raw_tools = raw.get("tools")
    if not isinstance(raw_tools, list) or not raw_tools:
        raise ValueError("Registry must contain a non-empty 'tools' list.")
    cleaned_tools: List[str] = []
    for t in raw_tools:
        if not isinstance(t, str) or not t.strip():
            raise ValueError(f"Tool names must be non-empty strings, got {t!r}")
        cleaned_tools.append(t.strip())
    if len(cleaned_tools) != len(set(cleaned_tools)):
        raise ValueError("Duplicate tool names found in 'tools' list.")

    tools_set = set(cleaned_tools)

    # 2. Validate type tokens
    raw_type_tokens = raw.get("type_tokens")
    if not isinstance(raw_type_tokens, dict) or not raw_type_tokens:
        raise ValueError("Registry must contain a non-empty 'type_tokens' dictionary.")

    # 3. Validate metrics
    raw_metrics = raw.get("metrics", {})
    metrics: Dict[str, MetricSpec] = {}
    for m_name, m_cfg in raw_metrics.items():
        kind = m_cfg.get("kind", "float")
        metrics[m_name] = MetricSpec(
            name=m_name,
            kind=kind,
            min_val=m_cfg.get("min"),
            max_val=m_cfg.get("max"),
        )

    # 4. Validate tool specs
    raw_tool_specs = raw.get("tool_specs", {})
    tool_specs: Dict[str, ToolSpec] = {}
    tool_return_map: Dict[str, List[str]] = {}

    for t_name, t_cfg in raw_tool_specs.items():
        if t_name not in tools_set:
            raise ValueError(f"Tool '{t_name}' in 'tool_specs' is not listed in 'tools'.")
        returns = t_cfg.get("returns", [])
        for ret in returns:
            if ret not in metrics:
                raise ValueError(
                    f"Tool '{t_name}' returns unregistered metric '{ret}'. "
                    f"Add '{ret}' to 'metrics' first."
                )
        scope = t_cfg.get("scope", "macro")
        if scope not in ("macro", "micro"):
            raise ValueError(f"Tool '{t_name}' has invalid scope '{scope}'. Must be 'macro' or 'micro'.")
        tool_specs[t_name] = ToolSpec(
            name=t_name,
            returns=list(returns),
            scope=scope,
            general=bool(t_cfg.get("general", False)),
        )
        tool_return_map[t_name] = list(returns)

    # Ensure all tools have at least a default ToolSpec
    for t in cleaned_tools:
        if t not in tool_specs:
            tool_specs[t] = ToolSpec(name=t, returns=[], scope="macro")
            tool_return_map[t] = []

    scope_schedule = list(raw.get("scope_schedule", ["macro", "micro"]))

    # 5. Validate domains
    raw_domains = raw.get("domains")
    if not isinstance(raw_domains, dict) or not raw_domains:
        raise ValueError("Registry must contain a non-empty 'domains' dictionary.")

    domains: Dict[str, DomainSpec] = {}
    for d_token, d_cfg in raw_domains.items():
        if d_token not in raw_type_tokens:
            raise ValueError(f"Domain key '{d_token}' is not present in 'type_tokens'.")

        instruction = d_cfg.get("instruction", "").strip()
        if not instruction:
            raise ValueError(f"Domain '{d_token}' has an empty instruction.")

        alert_metrics = list(d_cfg.get("alert_metrics", []))
        for am in alert_metrics:
            if am not in metrics:
                raise ValueError(
                    f"Domain '{d_token}' alert metric '{am}' is not defined in 'metrics'."
                )

        raw_hyps = d_cfg.get("hypotheses", [])
        if not isinstance(raw_hyps, list) or len(raw_hyps) < 2:
            raise ValueError(f"Domain '{d_token}' must have at least 2 hypotheses (including H_novel).")

        hypotheses: List[HypothesisSpec] = []
        for i, h_cfg in enumerate(raw_hyps):
            h_id = h_cfg.get("id", "").strip()
            if not h_id:
                raise ValueError(f"Domain '{d_token}' hypothesis at index {i} has empty 'id'.")
            h_text = h_cfg.get("text", "").strip()
            is_residual = bool(h_cfg.get("is_residual", False))
            h_risk = h_cfg.get("risk")
            if h_risk is not None:
                h_risk = float(h_risk)
                if not 0.0 <= h_risk <= 1.0:
                    raise ValueError(f"Hypothesis '{h_id}' in domain '{d_token}' risk must be in [0, 1], got {h_risk}.")

            # Associated tools
            assoc_tools = list(h_cfg.get("associated_tools", []))
            for at in assoc_tools:
                if at not in tools_set:
                    raise ValueError(
                        f"Hypothesis '{h_id}' in domain '{d_token}' references unknown tool '{at}'."
                    )

            # Rules
            raw_rules = h_cfg.get("falsification_rules", [])
            rules: List[FalsificationRule] = []

            if is_residual:
                if i != len(raw_hyps) - 1 or h_id != "H_novel":
                    raise ValueError(
                        f"Residual hypothesis must be 'H_novel' and located at the final index. "
                        f"Found id='{h_id}' at index {i} of {len(raw_hyps)}."
                    )
            else:
                if not raw_rules:
                    raise ValueError(
                        f"Known hypothesis '{h_id}' in domain '{d_token}' has no falsification rules "
                        f"and can never be eliminated."
                    )

            # Metrics observable by this hypothesis
            observable_metrics: Set[str] = set(alert_metrics)
            for at in assoc_tools:
                observable_metrics.update(tool_return_map.get(at, []))

            for r_cfg in raw_rules:
                r_metric = r_cfg.get("metric", "").strip()
                if not r_metric or r_metric not in metrics:
                    raise ValueError(
                        f"Rule in hypothesis '{h_id}' references unknown metric '{r_metric}'."
                    )
                if r_metric not in observable_metrics:
                    raise ValueError(
                        f"Rule metric '{r_metric}' in hypothesis '{h_id}' is neither in domain "
                        f"alert_metrics nor returned by associated tools {assoc_tools}."
                    )
                r_op = r_cfg.get("operator", "").strip()
                r_thresh = r_cfg.get("threshold")
                rules.append(FalsificationRule(metric=r_metric, operator=r_op, threshold=r_thresh))

            hypotheses.append(
                HypothesisSpec(
                    id=h_id,
                    text=h_text,
                    associated_tools=assoc_tools,
                    falsification_rules=rules,
                    is_residual=is_residual,
                    risk=h_risk,
                )
            )

        # Enforce last hypothesis is H_novel
        if hypotheses[-1].id != "H_novel" or not hypotheses[-1].is_residual:
            raise ValueError(
                f"Domain '{d_token}' must end with 'H_novel' as residual hypothesis."
            )

        d_risk = d_cfg.get("risk")
        if d_risk is not None:
            d_risk = float(d_risk)
            if not 0.0 <= d_risk <= 1.0:
                raise ValueError(f"Domain '{d_token}' risk must be in [0, 1], got {d_risk}.")

        domains[d_token] = DomainSpec(
            type_token=d_token,
            instruction=instruction,
            alert_metrics=alert_metrics,
            hypotheses=hypotheses,
            risk=d_risk,
        )

    return Registry(
        tools=cleaned_tools,
        type_tokens=raw_type_tokens,
        metrics=metrics,
        tool_specs=tool_specs,
        scope_schedule=scope_schedule,
        domains=domains,
        tool_hidden=raw.get("tool_hidden"),
        tool_dropout=float(raw.get("tool_dropout", 0.1)),
        tool_prior=float(raw.get("tool_prior")) if raw.get("tool_prior") is not None else None,
    )
