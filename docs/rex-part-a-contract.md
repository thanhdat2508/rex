# Rex Part A — Technical Interface Contract & Data Contracts

**Status:** Approved by Engineering Lead  
**Scope:** Part A (Unified Model, Ponder Block, 4 Heads, Hypothesis Scorer, InvestigatorAgent)  
**Collaborators:** B (Orchestration & Tool Runtime), C (Decision Adapter), F (Data & Joint Training), G (Telemetry & Ops)

---

## 1. Core Architecture Decisions

1. **Unified Model (`arch = "rex2"`):**
   - Single class `RexDecisionModel` in `rex/rex_model.py`.
   - Single forward execution point: `forward(state, evidence_new)` with alias `step(state, evidence_new)`.
   - Single checkpoint file. Legacy `rex_forward` is deprecated and preserved only as a lightweight 3-line compatibility shim.
   - Legacy `RexDecisionModel.investigate()` is eliminated from the model class; autonomous looping is handled by `InvestigatorAgent` in `rex/rex_agent.py`.

2. **1024-Dimensional ModernBERT Backbone:**
   - Pre-trained backbone from Laya (`convaiinnovations/laya`, 421M parameters).
   - Fast cross-attention with Key-Value caching across $T \in \{2..4\}$ pondering steps.

3. **Zero-Init Verdict Bridge:**
   - Logits for hypothesis $j$: $\text{logit}_j = \mathrm{Scorer}(m_j + g_v \cdot W_v h_T)$.
   - $W_v = 0$ and $g_v = 0$ at initialization. At checkpoint load time, verdict logits match legacy Laya checkpoints with zero numerical divergence ($\Delta < 10^{-7}$).

4. **Two Active Innovations:**
   - **Learned Early STALL:** Signal emitted when $S < 0.85$ and no allowed tool has expected information gain $\text{gain}_k \ge \tau_k$.
   - **Epistemic Tool Selection:** 3-branch dispatcher based on tool cost $\tau_k$, sufficiency $S$, and risk $R$. Top-1 blind dispatch is replaced.

---

## 2. Interface Contracts with Collaborators

### 2.1. Contract with Collaborator B (Orchestration & State Store)
- **Part A provides:**
  - `RexStepOutput` containing:
    - `selected_tools: List[str]` (approved tools for parallel dispatch under descending budget $K_t$).
    - `proposed_action: str` (`"CALL_TOOL"`, `"VERDICT"`, or `"STALL"`).
    - `stall: bool`, `stall_mode: str` (`"LEARNED"`), and `rule_stall: bool` (for telemetry comparison).
    - `to_trace_dict()`: Strictly serializable dictionary containing only primitive Python types (`float`, `int`, `str`, `bool`, `list`).
- **Part B provides:**
  - Tool catalogue with fields: `name`, `description`, `level` (`"READ"`, `"LIGHT"`, `"HEAVY"`), `cost` (used to derive $\tau_k$).
  - Execution of tool calls and returning structured observations.
  - Tracking of called tools in incident history to enforce non-repetition.

### 2.2. Contract with Collaborator C (Decision Adapter)
- **Part A provides:**
  - `verdict_probs: Dict[str, float]` across candidate hypotheses.
  - Temperature-scaled conformal prediction set `conformal_set: List[str]`.
  - Risk estimate `risk: float` ($p_R = \sigma(z_R)$).
- **Part C provides:**
  - Mapping from winning hypothesis label to business actions (`ALLOW`, `BLOCK`, `ESCALATE_HUMAN`).

### 2.3. Contract with Collaborator F (Joint Training & Checkpoints)
- **Part A provides:**
  - Partial checkpoint loading: `from_rex` loads base Laya weights (`encoder`, `head`, `scorer`, `type_emb`) while cleanly initializing Rex2 modules (`ponder_block`, `head_s`, `head_r`, `verdict_bridge`, `tool_matcher`, `tool_gain`, `hyp_scorer`).
  - Checkpoint metadata marked with `arch = "rex2"`.
  - Tool projection weight matrix $[N_{\text{tools}}, 256]$ matching `ToolOrthogonalLoss` requirements in `rex/losses.py`.
  - Pure label generation functions in `rex/rex_labels.py`.
- **Part F provides:**
  - Historical incident trajectories with unmasked observations and ground-truth key facts.
  - SFT and RL training of full backbone and joint objectives.

### 2.4. Contract with Collaborators D, E, G (Trace & Operations)
- Part A provides `to_trace_dict()` with the following verified telemetry keys:
  - `s_prob`: Sufficiency probability.
  - `r_prob`: Operational risk probability.
  - `top_tools`: List of top-3 candidate tools with relevance scores.
  - `tool_gains`: Dictionary mapping tool names to expected epistemic gain.
  - `verdict_probs`: Probability distribution over active hypotheses.
  - `alive_hypotheses`: List of hypotheses surviving Popperian elimination.
  - `stall`: Boolean flag.
  - `stall_mode`: String identifier (`"LEARNED"` or `"NONE"`).
  - `rule_stall`: Rule-based stagnation check ($\Delta S < \varepsilon$ across 2 rounds).
  - `gain_head_trained`: Boolean safety indicator.
  - `is_truncated`: Boolean indicating if evidence exceeded the 64-token cap.
