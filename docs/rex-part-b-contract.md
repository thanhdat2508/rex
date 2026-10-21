# Rex Part B — Technical Interface Contract & Orchestration Protocol

**Status:** Approved by Engineering Lead  
**Scope:** Part B (Orchestration, Tool Runtime, State Store, Safety Guard, Stop Conditions, Adapter)  
**Collaborators:** A (Unified Model & PoE), C (Decision Adapter), D/E (Reporting & Audit), F (Joint Training), G (Telemetry & Ops)

---

## 1. Core Responsibilities & Boundaries

Part B is the orchestration layer surrounding Rex: **Rex selects, B executes**.

1. **Tool Execution:**
   - Call tools selected by Rex via MCP Client over streamable HTTP / stdio.
   - Automatically extract invocation parameters from Case State (`state.<field>`) and prior round results (`results.<tool>.<metric>`).
   - Validate returned metrics against `MetricSpec` (bounds, type, presence).
   - Compress and pack multi-tool round observations into a bounded 64-token evidence chunk via `packer.py`.

2. **Access Control & Guardrail (Guard):**
   - 3-tier classification: `READ`, `LIGHT`, `HEAVY`.
   - In current milestone, **only READ tools are permitted** for automated execution.
   - Double-gating: Compute availability mask before inference; re-filter `selected_tools` after inference to guard against unmasked turn-1 / last-resort exploration in model.

3. **Termination & Escalation (Stop Engine):**
   - Pure, deterministic state machine evaluated after every agent step.
   - Outcome classification:
     - `AUTO_VERDICT`: $S \ge 0.85$ and action is `VERDICT` (without `HIGH_RISK`).
     - `HIGH_RISK`: $R \ge 0.30$ or rule floor tripped.
     - `DEADLOCK`: Model emits `STALL`, or $\Delta S < \varepsilon$ ($0.05$) across two consecutive rounds, or low-$S$ conformal verdict without approval.
     - `BUDGET_EXHAUSTED`: Exceeded round budget (3), call budget (6), or SLA deadline.
     - `TOOL_BLOCKED`: No callable tools remain or prohibited level requested.

4. **Persistence & Telemetry:**
   - In-memory `StateStore` for core engine; production persistent DB + lease lock in extra `rex[orch]`.
   - Emit strictly structured JSON traces for Collaborators D & E conforming to telemetry schemas.
   - Produce single conclusive handoff payload (`ticket_id:decision_seq`, `verdict_probs`, `conformal_set`, `risk`) for Collaborator C.

---

## 2. Interface Contract with Collaborator A (Model Engine)

### 2.1. APIs Consumed from Part A
- Turn 1: `encode_rex_input` $\to$ `collate_rex` $\to$ `RexDecisionModel.forward(...)` returning `(verdict_logits, state, step_out)`.
- Subsequent turns: `encode_evidence(tok, obs_text, round_idx, max_tokens=64)` $\to$ `RexDecisionModel.forward(state, evidence_new, ...)`.
- Hypothesis Scoring & Pruning: `model.hyp_scorer`, `HybridPoEEngine.evaluate`, `HardMaskingManager.compute_tool_mask`.
- Data Containers: `RexState`, `RexStepOutput`.

### 2.2. Guardrail Handling of Part A Behaviors
| Part A Behavior | Part B Handling & Mitigation |
| :--- | :--- |
| Missing `level` or `cost` in registry | Part B enriches `ToolSpec` from `config/rex_tools.yaml`. Fails startup if missing. |
| `rule_stall` always `False` in Part A telemetry | Part B tracks history of $S$ in `stop.py` and evaluates $\Delta S < 0.05$ across 2 rounds. |
| Model emits `VERDICT` when $S < 0.85$ | Governed by `accept_conformal_verdict` (default: `False`). Escalates as `DEADLOCK` (`CONFORMAL_LOW_S`). |
| Top-level Turn 1 / last-chance may bypass mask | `guard.py` filters `selected_tools` before dispatching to MCP. |
| Evidence length constraint | `packer.py` tokenizes formatted strings ensuring $\le 64$ tokens per round. |
| State serialization / crash recovery | Replay `forward` passes using stored evidence chain; restore `called_tools` from store. |

---

## 3. Configuration & Defaults (`config/rex_orchestrator.yaml`)

- Priority order when multiple escalation reasons trigger: `HIGH_RISK`, `DEADLOCK`, `BUDGET_EXHAUSTED`, `TOOL_BLOCKED`.
- `max_rounds`: 3
- `n_max_calls`: 6 ($K_t = 3, 2, 1$)
- $\varepsilon$ (stagnation threshold): 0.05
- Risk threshold: 0.30
- `accept_conformal_verdict`: `False` (safe default)
- Global case deadline: 300 ms
- Lease duration / heartbeat: 60s / 15s (in DB backend)
