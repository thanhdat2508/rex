"""Multi-task trajectory trainer for the Rex decision heads (T4, T10).

Each training item is one whole investigation case. Round 1 runs `forward(state=None)` on the
case's opening layout; every later round runs `forward(state, evidence_new=...)` on the gold
observations of the previous round, exactly like InvestigatorAgent / RexAdapter at inference.
Losses of every round are summed, so memory append, KV-cache extension and the thought state h
carried across rounds are all trained.

Stages (freeze schedule):
    heads  - train head_s, head_r, tool_matcher, tool_gain, hyp_scorer, alive_count_emb; the rest is frozen.
    ponder - additionally train ponder_block and verdict_bridge.
    full   - train every parameter; the encoder uses its own (smaller) learning rate.
"""
from __future__ import annotations

from collections import OrderedDict
import dataclasses
from dataclasses import dataclass, field
import gc
import math
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .calibrate import MIN_TYPE_N, fit_one_temperature
from .losses import RexLossWeights, RexMultiTaskInputs, RexMultiTaskLoss
from .rex_heads import compute_ece
from .rex_data import build_step_labels, collate_rex_train, format_state
from .rex_format import encode_evidence, encode_rex_input, format_observations
from .rex_tools import DEFAULT_R_THRESHOLD, DEFAULT_S_THRESHOLD, gain_threshold
from .rex_types import RexState

STAGES: Tuple[str, ...] = ("heads", "ponder", "full")
STAGE_MODULES: Dict[str, Tuple[str, ...]] = {
    "heads": ("head_s", "head_r", "tool_matcher", "tool_gain", "hyp_scorer", "alive_count_emb"),
    "ponder": ("head_s", "head_r", "tool_matcher", "tool_gain", "hyp_scorer", "alive_count_emb",
               "ponder_block", "verdict_bridge"),
}
MIN_CONFORMAL_CASES: int = 20


@dataclass
class RexTrainConfig:
    """Hyper-parameters of a Rex trajectory training run."""

    stage: str = "heads"
    epochs: int = 3
    batch_cases: int = 8
    grad_accum: int = 1
    lr_heads: float = 1e-4
    lr_encoder: float = 2.5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    max_len: int = 512
    head_max_len: int = 192
    evidence_max_tokens: int = 64
    shuffle_hypotheses: bool = True
    detach_between_rounds: bool = True
    amp: bool = False
    seed: int = 42
    log_every: int = 10
    loss_weights: RexLossWeights = field(default_factory=RexLossWeights)

    def validate(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {self.stage!r}")
        for name in ("epochs", "batch_cases", "grad_accum", "evidence_max_tokens"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError(f"warmup_ratio must be in [0, 1), got {self.warmup_ratio}")


@dataclass
class EncodedCase:
    """One case encoded for trajectory training: round-1 inputs plus per-round labels and evidence."""

    case_id: str
    round1: Dict[str, Any]
    rounds: List[Dict[str, Any]]
    evidence_ids: List[List[int]]
    evidence_truncated: List[bool]


def group_cases(samples: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Group step samples by case_id, keeping generation order inside each case."""
    cases: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    for s in samples:
        cases.setdefault(s["case_id"], []).append(s)
    return list(cases.values())


def encode_case(
    case: Sequence[Dict[str, Any]],
    tok: Any,
    tools: Sequence[str],
    registry: Any,
    rng: Optional[random.Random] = None,
    max_len: int = 512,
    head_max_len: int = 192,
    evidence_max_tokens: int = 64,
) -> EncodedCase:
    """Encode one case: the round-1 layout fixes the hypothesis order used by every round's labels."""
    first = case[0]
    domain = registry.get_domain(first["type_token"])
    hypotheses = domain.hypotheses_dict
    hyp_keys = list(hypotheses.keys())

    option_order = None
    ordered_keys = hyp_keys
    if rng is not None:
        option_order = list(range(len(hyp_keys)))
        rng.shuffle(option_order)
        ordered_keys = [hyp_keys[i] for i in option_order]

    round_labels = [build_step_labels(s, tools, ordered_keys, option_order=option_order) for s in case]

    encoded = encode_rex_input(
        tok=tok,
        type_token=first["type_token"],
        instruction=domain.instruction,
        hypotheses=hypotheses,
        state=format_state(first["state"]),
        alive=first["alive"],
        max_len=max_len,
        head_max_len=head_max_len,
        option_order=option_order,
        tool_mask=round_labels[0]["tool_mask"],
    )
    round1 = dict(round_labels[0])
    round1.update(input_ids=encoded["input_ids"], markers=encoded["markers"], hyp_alive=encoded["hyp_alive"])

    # Evidence entering round k (k >= 2) is the observation set gathered at step k - 1
    filler = _filler_token(tok)
    evidence_ids: List[List[int]] = []
    evidence_truncated: List[bool] = []
    for prev in case[:-1]:
        ev = encode_evidence(tok, format_observations(prev.get("observations", {})), max_tokens=evidence_max_tokens)
        evidence_ids.append(list(ev["input_ids"]) or [filler])
        evidence_truncated.append(bool(ev["is_truncated"]))

    return EncodedCase(
        case_id=first["case_id"],
        round1=round1,
        rounds=round_labels,
        evidence_ids=evidence_ids,
        evidence_truncated=evidence_truncated,
    )


def _filler_token(tok: Any) -> int:
    for name in ("sep_token_id", "eos_token_id", "pad_token_id"):
        tid = getattr(tok, name, None)
        if tid is not None:
            return int(tid)
    return 0


def _pad_id(tok: Any) -> int:
    return getattr(tok, "pad_token_id", 0) or 0


def _stack_round(rounds: Sequence[Dict[str, Any]], n_markers: int, n_tools: int) -> Dict[str, torch.Tensor]:
    """Label tensors for one round across a batch of cases (rows of finished cases are placeholders)."""
    b = len(rounds)
    out = {
        "tool_labels": torch.zeros((b, n_tools)),
        "gain_labels": torch.full((b, n_tools), -100.0),
        "redundancy_masks": torch.zeros((b, n_tools), dtype=torch.bool),
        "tool_mask": torch.ones((b, n_tools), dtype=torch.bool),
        "hyp_alive": torch.zeros((b, n_markers), dtype=torch.bool),
        "hyp_labels": torch.zeros((b, n_markers)),
        "hyp_risk_labels": torch.full((b, n_markers), -100.0),
        "target_verdict": torch.zeros(b, dtype=torch.long),
        "s_labels": torch.zeros((b, 1)),
        "r_labels": torch.zeros((b, 1)),
        "stall_labels": torch.zeros(b),
    }
    for i, r in enumerate(rounds):
        k = len(r["hyp_alive"])
        out["tool_labels"][i] = torch.tensor(r["tool_labels"])
        out["gain_labels"][i] = torch.tensor(r["gain_labels"])
        out["redundancy_masks"][i] = torch.tensor(r["redundancy_mask"], dtype=torch.bool)
        if r["tool_mask"] is not None:
            out["tool_mask"][i] = torch.tensor(r["tool_mask"], dtype=torch.bool)
        out["hyp_alive"][i, :k] = torch.tensor(r["hyp_alive"], dtype=torch.bool)
        out["hyp_labels"][i, :k] = torch.tensor(r["hyp_true"])
        if r.get("hyp_risk"):
            out["hyp_risk_labels"][i, :k] = torch.tensor(r["hyp_risk"])
        out["target_verdict"][i] = r["target_verdict"]
        out["s_labels"][i, 0] = r["s_label"]
        out["r_labels"][i, 0] = r["r_label"]
        out["stall_labels"][i] = r.get("stall_label", 0.0)
    # A finished case keeps one live slot so masked softmax/CE stay finite; its loss is dropped anyway
    dead = ~out["hyp_alive"].any(dim=1)
    out["hyp_alive"][dead, 0] = True
    return out


def _detach_state(state: RexState) -> RexState:
    kv = state.kv_cache
    return dataclasses.replace(
        state,
        h=state.h.detach(),
        M=state.M.detach(),
        hyp_markers=state.hyp_markers.detach() if state.hyp_markers is not None else None,
        kv_cache=(kv[0].detach(), kv[1].detach()) if kv is not None else None,
    )


def _step_loss(
    loss_fn: RexMultiTaskLoss,
    model: nn.Module,
    verdict_logits: torch.Tensor,
    out: Any,
    labels: Dict[str, torch.Tensor],
    marker_mask: torch.Tensor,
    rows: torch.Tensor,
):
    """Multi-task loss restricted to the batch rows whose trajectory is still running."""
    def pick(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return None if x is None else x.index_select(0, rows)

    return loss_fn(RexMultiTaskInputs(
        tool_logits=pick(out.tool_logits),
        tool_labels=pick(labels["tool_labels"]),
        verdict_logits=pick(verdict_logits),
        poe_mask=pick(labels["hyp_alive"].long()),
        target_verdict=pick(labels["target_verdict"]),
        tool_projection_weight=model.tool_matcher.tool_projection_weight,
        s_logits=pick(out.s_logit),
        s_targets=pick(labels["s_labels"]),
        r_logits=pick(out.r_logit),
        r_targets=pick(labels["r_labels"]),
        gain_logits=pick(out.tool_gain_logits),
        gain_targets=pick(labels["gain_labels"]),
        redundancy_mask=pick(labels["redundancy_masks"]),
        hyp_logits=pick(out.hyp_logits),
        hyp_targets=pick(labels["hyp_labels"]),
        hyp_mask=pick(marker_mask),
        hyp_risk_logits=pick(out.hyp_risk_logits),
        hyp_risk_targets=pick(labels["hyp_risk_labels"]),
    ))


def run_cases(
    model: nn.Module,
    cases: Sequence[EncodedCase],
    tok: Any,
    loss_fn: RexMultiTaskLoss,
    device: torch.device,
    detach_between_rounds: bool = True,
    autocast: Optional[Callable[[], Any]] = None,
    collect: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[str, torch.Tensor]]:
    """Forward a batch of cases round by round.

    Returns (mean loss over executed case-rounds, per-component sums for logging,
    final-round verdict probabilities and targets per case for calibration). When `collect` is a
    list, the head outputs and labels of every executed case-round are appended to it (evaluation).
    """
    n_tools = len(model.tools)
    batch1 = collate_rex_train([c.round1 for c in cases], _pad_id(tok))
    marker_mask = batch1["marker_mask"].to(device)
    n_markers = marker_mask.size(1)
    n_rounds = max(len(c.rounds) for c in cases)
    ctx = autocast if autocast is not None else _null_ctx

    total = torch.zeros((), device=device)
    n_steps = 0
    parts: Dict[str, float] = {}
    final_probs = torch.zeros((len(cases), n_markers))
    final_logits = torch.zeros((len(cases), n_markers))
    final_targets = torch.zeros(len(cases), dtype=torch.long)
    state: Optional[RexState] = None

    for k in range(n_rounds):
        active = [i for i, c in enumerate(cases) if k < len(c.rounds)]
        rows_k = [c.rounds[k] if k < len(c.rounds) else c.rounds[-1] for c in cases]
        labels = {n: t.to(device) for n, t in _stack_round(rows_k, n_markers, n_tools).items()}

        with ctx():
            if k == 0:
                verdict_logits, state, out = model(
                    input_ids=batch1["input_ids"].to(device),
                    attention_mask=batch1["attention_mask"].to(device),
                    marker_pos=batch1["marker_pos"].to(device),
                    marker_mask=marker_mask,
                    qtype=batch1["qtype"].to(device),
                    hyp_alive=labels["hyp_alive"],
                    tool_mask=labels["tool_mask"],
                    dispatch=False,
                )
            else:
                if detach_between_rounds:
                    state = _detach_state(state)
                filler = _filler_token(tok)
                evs = [c.evidence_ids[k - 1] if k - 1 < len(c.evidence_ids) else [filler] for c in cases]
                width = max(len(e) for e in evs)
                ev_ids = torch.full((len(cases), width), _pad_id(tok), dtype=torch.long)
                ev_mask = torch.zeros((len(cases), width), dtype=torch.bool)
                for i, e in enumerate(evs):
                    ev_ids[i, : len(e)] = torch.tensor(e, dtype=torch.long)
                    ev_mask[i, : len(e)] = True
                truncated = any(
                    c.evidence_truncated[k - 1] for c in cases if k - 1 < len(c.evidence_truncated)
                )
                verdict_logits, state, out = model(
                    state=state,
                    evidence_new=ev_ids.to(device),
                    evidence_mask=ev_mask.to(device),
                    hyp_alive=labels["hyp_alive"],
                    tool_mask=labels["tool_mask"],
                    is_truncated=truncated,
                    dispatch=False,
                )

        rows = torch.tensor(active, dtype=torch.long, device=device)
        lb = _step_loss(loss_fn, model, verdict_logits, out, labels, marker_mask, rows)
        total = total + lb.total_loss * len(active)
        n_steps += len(active)
        for name in ("tool_loss", "verdict_loss", "ortho_loss", "s_loss", "r_loss", "gain_loss", "hyp_loss",
                     "hyp_risk_loss"):
            v = getattr(lb, name)
            if v is not None:
                parts[name] = parts.get(name, 0.0) + float(v.detach()) * len(active)

        if collect is not None:
            sel = rows.cpu()
            collect.append({
                "round": torch.full((len(active),), k),
                "s_prob": out.s_prob.detach().float().cpu().view(-1)[sel],
                "s_label": labels["s_labels"].cpu().view(-1)[sel],
                "r_prob": out.r_prob.detach().float().cpu().view(-1)[sel],
                "r_label": labels["r_labels"].cpu().view(-1)[sel],
                "gain": out.tool_gains.detach().float().cpu()[sel],
                "gain_label": labels["gain_labels"].cpu()[sel],
                "callable": (labels["tool_mask"] & ~labels["redundancy_masks"]).cpu()[sel],
                "hyp": out.hyp_scores.detach().float().cpu()[sel][marker_mask.cpu()[sel]],
                "hyp_label": labels["hyp_labels"].cpu()[sel][marker_mask.cpu()[sel]],
                "stall_label": labels["stall_labels"].cpu()[sel],
            })

        logits_k = verdict_logits.detach().float().cpu()
        probs = out.verdict_probs.detach().float().cpu()
        for i in active:
            if k == len(cases[i].rounds) - 1:
                final_probs[i] = probs[i]
                final_logits[i] = logits_k[i]
                final_targets[i] = int(labels["target_verdict"][i])

    parts = {name: v / max(1, n_steps) for name, v in parts.items()}
    return total / max(1, n_steps), parts, {"probs": final_probs, "logits": final_logits,
                                            "targets": final_targets, "marker_mask": marker_mask.cpu()}


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def configure_stage(model: nn.Module, stage: str) -> None:
    """Freeze/unfreeze parameters for a training stage."""
    if stage == "full":
        for p in model.parameters():
            p.requires_grad_(True)
        return
    trainable = STAGE_MODULES[stage]
    for name, p in model.named_parameters():
        p.requires_grad_(name.split(".", 1)[0] in trainable)


def _set_train_mode(model: nn.Module, stage: str) -> None:
    model.train()
    if stage != "full":
        # Frozen modules run deterministically (no dropout) while the trained heads see dropout
        trainable = set(STAGE_MODULES[stage])
        for name, child in model.named_children():
            if name not in trainable:
                child.eval()


def build_optimizer(model: nn.Module, config: RexTrainConfig) -> torch.optim.Optimizer:
    enc_ids = {id(p) for p in model.encoder.parameters()}
    enc = [p for p in model.parameters() if p.requires_grad and id(p) in enc_ids]
    rest = [p for p in model.parameters() if p.requires_grad and id(p) not in enc_ids]
    groups = [{"params": rest, "lr": config.lr_heads}]
    if enc:
        groups.append({"params": enc, "lr": config.lr_encoder})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def _lr_lambda(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return fn


def grad_norms_by_module(model: nn.Module) -> Dict[str, float]:
    """L2 gradient norm per top-level module (0.0 when no parameter of it received a gradient)."""
    norms: Dict[str, float] = {}
    for name, p in model.named_parameters():
        top = name.split(".", 1)[0]
        g = 0.0 if p.grad is None else float(p.grad.detach().float().pow(2).sum())
        norms[top] = norms.get(top, 0.0) + g
    return {k: math.sqrt(v) for k, v in norms.items()}


def _autocast_factory(device: torch.device, amp: bool):
    if not amp:
        return None, None
    dev_type = device.type
    if dev_type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if dtype == torch.float16:
            scaler = getattr(torch.amp, "GradScaler", torch.cuda.amp.GradScaler)("cuda")
        else:
            scaler = None
        return (lambda: torch.autocast("cuda", dtype=dtype)), scaler
    elif dev_type == "cpu":
        return (lambda: torch.autocast("cpu", dtype=torch.bfloat16)), None
    elif dev_type == "mps":
        return (lambda: torch.autocast("mps", dtype=torch.float16)), None
    return None, None


def auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Area under the ROC curve (Mann-Whitney, ties count half); NaN when only one class is present."""
    scores, labels = scores.float().view(-1), labels.view(-1) > 0.5
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=scores.dtype)
    for v in torch.unique(scores):  # average ranks of tied scores
        tie = scores == v
        if int(tie.sum()) > 1:
            ranks[tie] = ranks[tie].mean()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def head_metrics(steps: Sequence[Dict[str, torch.Tensor]], tool_levels: Sequence[str],
                 tool_costs: Sequence[float]) -> Dict[str, float]:
    """Quality of the S, R, gain and hypothesis heads and of the derived STALL decision."""
    if not steps:
        return {}
    cat = {k: torch.cat([s[k] for s in steps]) for k in steps[0]}
    m: Dict[str, float] = {}

    s_prob, s_label, rnd = cat["s_prob"], cat["s_label"], cat["round"]
    m["s_auroc"] = auroc(s_prob, s_label)
    m["s_ece"] = compute_ece(s_prob, s_label)
    # Within-round AUROC: a head that only counts rounds scores 0.5 here even if its pooled AUROC is high
    per_round = [(auroc(s_prob[rnd == k], s_label[rnd == k]), int((rnd == k).sum())) for k in torch.unique(rnd)]
    per_round = [(a, n) for a, n in per_round if a == a]
    m["s_auroc_within_round"] = (sum(a * n for a, n in per_round) / sum(n for _, n in per_round)
                                 if per_round else float("nan"))
    for k in torch.unique(rnd).tolist():
        m[f"s_mean_round_{int(k) + 1}"] = float(s_prob[rnd == k].mean())

    r_prob, r_label = cat["r_prob"], cat["r_label"]
    m["r_mae"] = float((r_prob - r_label).abs().mean())
    m["r_auroc_high_risk"] = auroc(r_prob, (r_label > DEFAULT_R_THRESHOLD).float())

    gain, g_label, callable_ = cat["gain"], cat["gain_label"], cat["callable"]
    valid = callable_ & (g_label != -100)
    m["gain_auroc"] = auroc(gain[valid], g_label[valid])

    m["hyp_auroc"] = auroc(cat["hyp"], cat["hyp_label"])

    # STALL as the dispatcher derives it: not sufficient and no callable tool reaches its threshold
    tau = torch.tensor([gain_threshold(lv, c) for lv, c in zip(tool_levels, tool_costs)])
    any_tool = ((gain >= tau) & callable_).any(dim=1)
    pred_stall = (s_prob < DEFAULT_S_THRESHOLD) & ~any_tool
    pred_stall_high_risk = (s_prob >= DEFAULT_S_THRESHOLD) & (r_prob > DEFAULT_R_THRESHOLD)
    true_stall = cat["stall_label"] > 0.5
    m["stall_rate_label"] = float(true_stall.float().mean())
    m["stall_rate_pred"] = float(pred_stall.float().mean())
    m["stall_high_risk_rate_pred"] = float(pred_stall_high_risk.float().mean())
    m["stall_recall"] = float((pred_stall & true_stall).sum() / true_stall.sum().clamp(min=1))
    m["stall_precision"] = float((pred_stall & true_stall).sum() / pred_stall.sum().clamp(min=1))
    return m


def evaluate(
    model: nn.Module,
    cases: Sequence[EncodedCase],
    tok: Any,
    config: RexTrainConfig,
    device: torch.device,
) -> Dict[str, Any]:
    """Validation loss, final-round verdict accuracy, head metrics (see head_metrics) and conformal coverage.

    Also returns the final-round verdict probabilities/targets (conformal calibration) and
    per-case (live logits, one-hot target) pairs (temperature fitting).
    """
    loss_fn = RexMultiTaskLoss(weights=config.loss_weights)
    model.eval()
    probs_all, targets_all = [], []
    pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    steps: List[Dict[str, torch.Tensor]] = []
    total, n = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(cases), config.batch_cases):
            chunk = cases[start:start + config.batch_cases]
            loss, _, final = run_cases(model, chunk, tok, loss_fn, device, detach_between_rounds=True,
                                       collect=steps)
            total += float(loss) * len(chunk)
            n += len(chunk)
            probs_all.append(final["probs"])
            targets_all.append(final["targets"])
            for i in range(len(chunk)):
                live = final["marker_mask"][i] & (final["logits"][i] > -1e3)
                idx = live.nonzero(as_tuple=True)[0]
                target = int(final["targets"][i])
                if target in idx.tolist():
                    onehot = (idx == target).float()
                    pairs.append((final["logits"][i, idx], onehot))
    if not probs_all:
        return {"loss": float("nan"), "verdict_acc": float("nan"), "probs": None, "targets": None, "pairs": []}
    width = max(p.size(1) for p in probs_all)
    probs = torch.cat([torch.nn.functional.pad(p, (0, width - p.size(1))) for p in probs_all])
    targets = torch.cat(targets_all)
    acc = float((probs.argmax(dim=-1) == targets).float().mean())
    levels = [model.tool_meta.get(t, {}).get("level", "READ") for t in model.tools]
    costs = [float(model.tool_meta.get(t, {}).get("cost", 0.0)) for t in model.tools]
    metrics = head_metrics(steps, levels, costs)
    sets = model.calibrator.predict_set(probs, q_hat=model.conformal_threshold())
    names = [f"H_{i}" for i in range(probs.size(1))]
    live = (probs > 0).sum(dim=1).clamp(min=1).float()
    metrics["conformal_coverage"] = float(sum(names[int(t)] in st for st, t in zip(sets, targets)) / len(sets))
    metrics["conformal_mean_set_size"] = float(sum(len(st) for st in sets) / len(sets))
    metrics["conformal_mean_live_hypotheses"] = float(live.mean())
    return {"loss": total / max(1, n), "verdict_acc": acc, "probs": probs, "targets": targets, "pairs": pairs,
            "metrics": metrics}


def train_rex(
    model: nn.Module,
    tok: Any,
    registry: Any,
    train_samples: Sequence[Dict[str, Any]],
    val_samples: Optional[Sequence[Dict[str, Any]]] = None,
    config: Optional[RexTrainConfig] = None,
    calib_samples: Optional[Sequence[Dict[str, Any]]] = None,
    device: Optional[torch.device] = None,
    encode_tools: bool = True,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Train a RexDecisionModel in place on trajectory samples produced by rex_data.generate().

    Encodes the registry tool descriptions into the model (T5) unless `encode_tools=False`,
    marks the gain head trained when the gain loss is active, and fits the verdict temperature and
    the conformal threshold on `calib_samples`. Without them, the validation cases are split by case:
    half calibrates, the other half is the reported evaluation, so no metric is measured on the cases
    that fitted the calibration. Returns a summary with per-epoch losses and val metrics.
    """
    config = config or RexTrainConfig()
    config.validate()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    model.to(device)

    registry.check_model_tools(model.tools)
    model.set_tool_specs([registry.tool_specs[t] for t in model.tools])
    if encode_tools:
        model.encode_tool_descriptions([registry.tool_specs[t] for t in model.tools], tok)

    train_cases = group_cases(train_samples)
    val_cases_raw = group_cases(val_samples or [])
    if not train_cases:
        raise ValueError("train_samples is empty")

    def encode_all(cases, epoch: int, shuffle: bool) -> List[EncodedCase]:
        out = []
        for idx, case in enumerate(cases):
            rng = random.Random(config.seed + epoch * 1000003 + idx) if shuffle else None
            out.append(encode_case(case, tok, model.tools, registry, rng=rng, max_len=config.max_len,
                                   head_max_len=config.head_max_len,
                                   evidence_max_tokens=config.evidence_max_tokens))
        return out

    calib_cases_raw = group_cases(calib_samples or [])
    if not calib_cases_raw and len(val_cases_raw) >= 2:
        order = list(range(len(val_cases_raw)))
        random.Random(config.seed).shuffle(order)
        half = len(order) // 2
        calib_cases_raw = [val_cases_raw[i] for i in sorted(order[:half])]
        val_cases_raw = [val_cases_raw[i] for i in sorted(order[half:])]
    val_cases = encode_all(val_cases_raw, 0, shuffle=False)
    calib_cases = encode_all(calib_cases_raw, 0, shuffle=False)

    configure_stage(model, config.stage)
    optimizer = build_optimizer(model, config)
    micro_steps = math.ceil(len(train_cases) / config.batch_cases)
    total_updates = math.ceil(micro_steps / config.grad_accum) * config.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(total_updates, int(total_updates * config.warmup_ratio))
    )
    autocast, scaler = _autocast_factory(device, config.amp)
    loss_fn = RexMultiTaskLoss(weights=config.loss_weights)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"rex train: stage={config.stage} cases={len(train_cases)} calib_cases={len(calib_cases)} "
        f"val_cases={len(val_cases)} "
        f"trainable_params={n_trainable:,} updates={total_updates}")

    epoch_losses: List[float] = []
    first_grad_norms: Optional[Dict[str, float]] = None
    update = 0
    for epoch in range(config.epochs):
        _set_train_mode(model, config.stage)
        encoded = encode_all(train_cases, epoch, shuffle=config.shuffle_hypotheses)
        order = list(range(len(encoded)))
        random.Random(config.seed + epoch).shuffle(order)
        running, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for step in range(micro_steps):
            chunk = [encoded[i] for i in order[step * config.batch_cases:(step + 1) * config.batch_cases]]
            loss, parts, _ = run_cases(model, chunk, tok, loss_fn, device,
                                       detach_between_rounds=config.detach_between_rounds, autocast=autocast)
            scaled = loss / config.grad_accum
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            running += float(loss.detach()) * len(chunk)
            seen += len(chunk)

            last_micro = (step + 1) % config.grad_accum == 0 or step == micro_steps - 1
            if not last_micro:
                continue
            if scaler is not None:
                scaler.unscale_(optimizer)
            if first_grad_norms is None:
                first_grad_norms = grad_norms_by_module(model)
                log("rex train: first-update grad norms " + ", ".join(
                    f"{k}={v:.3g}" for k, v in sorted(first_grad_norms.items())))
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           config.max_grad_norm)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            if config.log_every and update % config.log_every == 0:
                log(f"rex train: epoch {epoch + 1} update {update}/{total_updates} loss {running / seen:.4f} "
                    + " ".join(f"{k}={v:.3f}" for k, v in parts.items()))
        del encoded, order
        gc.collect()
        epoch_losses.append(running / max(1, seen))
        log(f"rex train: epoch {epoch + 1} mean loss {epoch_losses[-1]:.4f}")

    if config.loss_weights.gain_weight > 0:
        model.mark_gain_head_trained(True)

    summary: Dict[str, Any] = {"epoch_losses": epoch_losses, "grad_norms": first_grad_norms or {}}
    summary["calib_cases"] = len(calib_cases)
    summary["val_cases"] = len(val_cases)
    if calib_cases:
        calib = evaluate(model, calib_cases, tok, config, device)
        if len(calib["pairs"]) >= MIN_TYPE_N:
            # Verdict temperature first, so the conformal threshold is fitted on calibrated probabilities
            temp = fit_one_temperature(calib["pairs"], min_n=MIN_TYPE_N)
            model.set_verdict_temperature(temp)
            summary["verdict_temperature"] = temp
            log(f"rex train: verdict temperature {temp:.3f} on {len(calib['pairs'])} calibration cases")
            calib = evaluate(model, calib_cases, tok, config, device)
        if len(calib_cases) >= MIN_CONFORMAL_CASES:
            summary["q_hat"] = model.calibrate_conformal(calib["probs"], calib["targets"])
            log(f"rex train: conformal q_hat {summary['q_hat']:.4f} on {len(calib_cases)} calibration cases")
        else:
            log(f"rex train: conformal threshold not fitted: {len(calib_cases)} calibration cases, "
                f"fewer than {MIN_CONFORMAL_CASES}")
    if val_cases:
        metrics = evaluate(model, val_cases, tok, config, device)
        summary["val_loss"] = metrics["loss"]
        summary["val_verdict_acc"] = metrics["verdict_acc"]
        summary["val_metrics"] = metrics["metrics"]
        log(f"rex train: val loss {metrics['loss']:.4f} final-round verdict acc {metrics['verdict_acc']:.3f}")
        log("rex train: val " + " ".join(f"{k}={v:.3f}" for k, v in metrics["metrics"].items()))
    del val_cases, calib_cases, train_cases, optimizer, scheduler, loss_fn
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.eval()
    return summary
