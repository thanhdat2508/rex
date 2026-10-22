# Rex Part A — Technical Interface Contract & Data Contracts

**Status:** Approved by Engineering Lead  
**Scope:** Part A (Unified Model, Ponder Block, 4 Heads, Hypothesis Scorer, InvestigatorAgent)  
**Collaborators:** B (Orchestration & Tool Runtime), C (Decision Adapter), F (Data & Joint Training), G (Telemetry & Ops)

---

## 1. Core Architecture Decisions

1. **Unified Model (`arch = "rex2"`):**
   - Single class `RexDecisionModel` in `rex/rex_model.py`.
   - Single forward execution point: `forward(state, evidence_new)` with alias `step(state, evidence_new)`.
   - The sufficiency head reads $h_T$ plus a learned embedding of the number of surviving cataloged hypotheses (`alive_count_emb`, zero-initialized): $S^*$ is defined on that survival set, which the model knows from the PoE mask.
   - `ingest_evidence(state, evidence_new)` appends evidence to memory without pondering, so callers score hypotheses with `hyp_scorer` on memory that already holds the newest observations, then call `forward(state)`.
   - Single checkpoint file. Legacy `rex_forward` is deprecated and preserved only as a lightweight 3-line compatibility shim.
   - Legacy `RexDecisionModel.investigate()` is eliminated from the model class; autonomous looping is handled by `InvestigatorAgent` in `rex/rex_agent.py`.

2. **1024-Dimensional ModernBERT Backbone:**
   - Pre-trained backbone from Laya (`convaiinnovations/laya`, 421M parameters).
   - Fast cross-attention with Key-Value caching across $T \in \{2..4\}$ pondering steps.

3. **Zero-Init Verdict Bridge:**
   - Logits for hypothesis $j$: $\text{logit}_j = \mathrm{Scorer}(m_j + g_v \cdot W_v h_T)$.
   - $g_v = 0$ and $W_v \sim \mathcal{N}(0, 0.02^2)$ at initialization, so the bridge adds exactly zero and verdict logits match legacy Laya checkpoints ($\Delta < 10^{-7}$), while $\partial L / \partial g_v \neq 0$ lets the bridge learn (both factors at zero would keep both gradients at zero).

4. **Two Active Innovations:**
   - **Learned Early STALL:** Signal emitted when $S < 0.85$ and no allowed tool has expected information gain $\text{gain}_k \ge \tau_k$.
   - **Epistemic Tool Selection:** 3-branch dispatcher based on tool cost $\tau_k$, sufficiency $S$, and risk $R$. Top-1 blind dispatch is replaced.
   - Gain labels are binary (`docs/rex-label-schema.md`), so $\text{gain}_k$ is the predicted probability that tool $k$ uncovers a key fact and $\tau_k$ is a probability threshold (`rex_tools.gain_threshold`): a $0.05$ floor for cheap READ tools, $\max(0.05, \min(0.5, \text{cost}))$ otherwise. The gain of an uncalled tool depends on values not yet observed, so the predicted gain stays near the base rate and mainly ranks tools; the budget $K_t$ bounds the calls.
   - The dispatcher checks sufficiency first: $S \ge 0.85$ and $R \le \tau_R$ proposes `VERDICT`; $S \ge 0.85$ and $R > \tau_R$ proposes `STALL` (`stall_reason = "HIGH_RISK"`), since more tools cannot lower business risk.

---

## 2. Interface Contracts with Collaborators

### 2.1. Contract with Collaborator B (Orchestration & State Store)
- **Part A provides:**
  - `RexStepOutput` containing:
    - `selected_tools: List[str]` (approved tools for parallel dispatch under descending budget $K_t$).
    - `proposed_action: str` (`"CALL_TOOL"`, `"VERDICT"`, or `"STALL"`).
    - `stall: bool`, `stall_mode: str` (`"LEARNED"`), and `rule_stall: bool` (for telemetry comparison).
    - `stall_reason: str` (`"NO_GAIN"` or `"HIGH_RISK"` for a STALL, `"NONE"` otherwise).
    - `to_trace_dict()`: Strictly serializable dictionary containing only primitive Python types (`float`, `int`, `str`, `bool`, `list`).
- **Part B provides:**
  - Tool catalogue with fields: `name`, `description`, `level` (`"READ"`, `"LIGHT"`, `"HEAVY"`), `cost` (used to derive $\tau_k$).
  - Execution of tool calls and returning structured observations.
  - Tracking of called tools in incident history to enforce non-repetition.

### 2.2. Contract with Collaborator C (Decision Adapter)
- **Part A provides:**
  - `verdict_probs: Dict[str, float]` across candidate hypotheses.
  - Temperature-scaled conformal prediction set `conformal_set: List[str]`.
  - Risk estimate `risk: float`, factorized over hypotheses: $p_R = \sum_j p_j \, \sigma(z_{R,j})$, where $z_{R,j}$ is the risk head applied to the bridged marker of hypothesis $j$ (the risk of $j$ being the true cause, `hyp_risk`) and $p_j$ the verdict probability. It therefore follows the evidence instead of the case type.
- **Part C provides:**
  - Mapping from winning hypothesis label to business actions (`ALLOW`, `BLOCK`, `ESCALATE_HUMAN`).

### 2.3. Contract with Collaborator F (Joint Training & Checkpoints)
- **Part A provides:**
  - Partial checkpoint loading: `from_rex` loads base Laya weights (`encoder`, `head`, `scorer`, `type_emb`) while cleanly initializing Rex2 modules (`ponder_block`, `head_s`, `head_r`, `verdict_bridge`, `tool_matcher`, `tool_gain`, `hyp_scorer`).
  - Checkpoint metadata marked with `arch = "rex2"`.
  - Tool projection weight matrix $[N_{\text{tools}}, 256]$ matching `ToolOrthogonalLoss` requirements in `rex/losses.py`.
  - Pure label generation functions in `rex/rex_labels.py`.
  - Trajectory trainer `rex/rex_train.py` (CLI `scripts/train_rex.py`): round 1 runs `forward(state=None)`, later rounds `forward(state, evidence_new)` on gold observations, with freeze stages `heads` → `ponder` → `full`. It encodes registry tool descriptions into `tool_embeddings`, sets `gain_head_trained`, and stores the conformal `q_hat` in the checkpoint (`calibrator_q_hat`).
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
  - `stall_reason`: `"NO_GAIN"`, `"HIGH_RISK"` or `"NONE"`.
  - `rule_stall`: Rule-based stagnation check ($\Delta S < \varepsilon$ across 2 rounds).
  - `gain_head_trained`: Boolean safety indicator.
  - `is_truncated`: Boolean indicating if evidence exceeded the 64-token cap.
