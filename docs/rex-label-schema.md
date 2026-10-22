# Rex Training Label & Dataset Schema (Collaborator F)

This document formalizes the JSON schema for multi-turn training samples and counterfactual supervision labels used to train the Rex unified architecture. It describes the records `rex.rex_data.generate()` and `scripts/generate_rex_data.py` write, which `rex.rex_train` and `RexTrajectoryDataset` read.

---

## 1. Top-Level Sample Format (JSON Lines record)

Each row in `train.jsonl` or `val.jsonl` represents a single decision step $(case, round)$ within an incident exploration trajectory. The rows of one case are consecutive and in round order; the trajectory trainer groups them by `case_id`.

```json
{
  "case_id": "case-000042",
  "t": 2,
  "type_token": "<TYPE_PAYMENT_FRAUD>",
  "state": {
    "alert": {"txn_count_1h": 41, "decline_ratio": 0.62},
    "round_1": {"transaction_velocity": {"small_amount_ratio": 0.91}}
  },
  "alive": {"H_card_testing": true, "H_compromised_merchant": false, "H_novel": true},
  "allowed_tools": ["payment_graph", "device_fingerprint"],
  "tool_labels": ["payment_graph"],
  "called_tools": ["payment_graph", "device_fingerprint"],
  "label_scope": "micro",
  "observations": {"payment_graph": {"shared_merchant_ratio": 0.08}},
  "target": "H_card_testing",
  "eliminated_by": {"H_compromised_merchant": ["merchant_dispute_rate LT 0.1 (observed 0.03)"]},
  "labels": {
    "s_star": 0.0,
    "r_star": 0.25,
    "gain_star": {"payment_graph": 1.0, "device_fingerprint": 0.0, "auth_history": -100.0},
    "stall_star": 0.0,
    "redundancy_mask": [true, false, false, true],
    "hyp_true": {"H_card_testing": 1.0, "H_compromised_merchant": 0.0, "H_novel": 1.0},
    "hyp_risk": {"H_card_testing": 0.25, "H_compromised_merchant": 0.7, "H_novel": 0.7}
  }
}
```

---

## 2. Field Definitions

| Field | Type | Description |
| :--- | :--- | :--- |
| `case_id` | `str` | Unique incident identifier. |
| `t` | `int` | Turn index (1, 2, 3). A case ends with a terminal step (`tool_labels == []`); when all three rounds called tools, that terminal step also carries `t = 3`. |
| `type_token` | `str` | Domain type token; selects the registry domain (instruction, hypotheses, rules). |
| `state` | `dict` | Evidence before this step: `alert` metrics plus `round_k` observations of earlier rounds. Round-1 input text is `format_state(state)`. |
| `alive` | `dict[str, bool]` | Survival after Popperian elimination on everything in `state` (alert included). `H_novel` is always alive. |
| `allowed_tools` | `list[str]` | Tools still callable: not called yet and serving at least one surviving hypothesis. |
| `tool_labels` | `list[str]` | Teacher tools for this round (`labels.teacher_tools` in the original schema). The default teacher is privileged: it only picks allowed tools with `gain_star == 1`, by scope schedule and round budget $K_t = (3, 2, 1)$; empty when no tool is worth calling. |
| `called_tools` | `list[str]` | What the behaviour policy called this round: the teacher tools first, then exploratory tools up to the budget. It decides the next state, so training states include rounds that uncover nothing, as at inference. The case ends when the behaviour policy has nothing left to call. |
| `label_scope` | `str` | Scope of the teacher tools (`macro`, `micro`, `mixed`), `none` (tools are called but none is worth it) or `terminal`. |
| `observations` | `dict` | What `called_tools` returned, grouped per tool; the evidence of the next round. The trainer and InvestigatorAgent render it with `rex_format.format_observations`, which matches the Part B packer (`tool OK k=v ... \| tool OK ...`). |
| `target` | `str` | The true hypothesis of the case. |
| `eliminated_by` | `dict[str, list[str]]` | Proof trace: for each eliminated hypothesis, the rules that falsified it and the observed values. |
| `labels.s_star` | `float (0/1)` | Sufficiency (`sufficient`): 1 when the target is the only surviving cataloged hypothesis, or when every cataloged hypothesis is eliminated and the target is `H_novel`. `H_novel` is not counted as a competing survivor. |
| `labels.r_star` | `float` | Business risk (`risk_label`) in $[0, 1]$ of acting on the true cause: the target hypothesis' registry `risk`, else the domain `risk`, else `rex_labels.DEFAULT_RISK_MAPPING`. It varies with the case, so the risk head learns the expected risk given the evidence. |
| `labels.gain_star` | `dict[str, float]` | Counterfactual gain (`tool_gains`) per tool: `1` if the tool's observations would falsify a surviving cataloged hypothesis (it uncovers an unseen key fact), `0` otherwise, `-100` if none of its metrics was recorded (masked from the loss). |
| `labels.stall_star` | `float (0/1)` | `is_stall`: `1` if $S^* = 0$ and no allowed tool has `gain_star == 1`. Evaluation target only: STALL is derived by the dispatcher from $S$, the gains and the remaining tools, so there is no STALL loss; `rex_train.evaluate` reports how often the derived STALL matches it. |
| `labels.redundancy_mask` | `list[bool]` | Per model tool: already called, or every hypothesis it serves is eliminated. Masked out of the gain loss. |
| `labels.hyp_true` | `dict[str, float]` | Survival target per hypothesis for `HypothesisScorer` (1 alive, 0 eliminated). |
| `labels.hyp_risk` | `dict[str, float]` | Registry risk of every hypothesis of the domain (domain risk when it sets none): the target of the per-hypothesis risk $r_j$, whose verdict-weighted sum is the reported risk $R$. |

The target verdict index is not stored: `rex_data.build_step_labels` derives it from `target` in the hypothesis order of the case's round-1 layout, so shuffling hypotheses keeps labels and slots aligned.

### Names used by the original draft

| Original draft | Record field |
| :--- | :--- |
| `domain_key` | `type_token` |
| `round_idx` | `t` |
| `history`, `new_evidence` | `state`, `observations` of the previous step |
| `surviving_hypotheses` | `alive` |
| `labels.sufficient` | `labels.s_star` |
| `labels.target_verdict` | derived from `target` |
| `labels.risk_label` | `labels.r_star` |
| `labels.tool_gains` | `labels.gain_star` |
| `labels.teacher_tools` | `tool_labels` |
| `labels.is_stall` | `labels.stall_star` |
| `labels.key_facts_remaining` | not produced; `gain_star` encodes which tools still uncover a key fact |
