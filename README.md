<div align="center">

# Rex Decision Engine

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Rex2%20%7C%20PoE%20%7C%20MECE-success.svg)](#neural-architecture-dual-head-projections)
[![SLA](https://img.shields.io/badge/SLA-Sub--Second%20%28%3C1s%29-brightgreen.svg)](#operational-slas-performance-benchmarks)
[![Security](https://img.shields.io/badge/VPC%20Security-Zero%20Inbound%20Ports-orange.svg)](#4-enterprise-security-zero-inbound-ports)

**Autonomous, Non-Autoregressive Popperian Decision & Multi-Turn Investigation Engine**  
_Built for Risk-Based Authentication (RBA), Account Takeover (ATO) Prevention, Sybil Defense, and Real-Time Fraud Incident Investigation._

[Key Innovations](#key-architectural-pillars) • [Neural Architecture](#neural-architecture-dual-head-projections) • [Multi-Turn Loop](#multi-turn-closed-loop-investigation-cycle) • [Quickstart](#quickstart) • [WAF Backtesting](#action-dispatch-incident-remediation) • [Training & Loss](#multi-task-training-loss-dynamics)

</div>

---

## Executive Overview

**Rex** is a production-grade, non-autoregressive decision and investigation engine engineered to resolve complex security alerts in **sub-second latency (< 1.0 s)**.

Traditional fraud defense systems suffer from a fundamental trade-off:

- **Heuristic rule engines & static classifiers** generate brittle alert floods, fail on multi-hop sybil syndicates, and trigger catastrophic false positives on shared network infrastructure (such as mobile carrier CGNAT).
- **Autoregressive Large Language Models (LLMs)** are too slow (3–10 s per turn), induce non-deterministic hallucinations, and cannot provide mathematically grounded risk bounds required for enterprise compliance.

Rex overcomes these limitations by combining a high-throughput **Transformer Encoder backbone** (e.g., ModernBERT) with **Karl Popper’s Process of Elimination (PoE)** and a **Mutually Exclusive, Collectively Exhaustive (MECE)** hypothesis taxonomy. Rather than attempting to "confirm" suspicion through superficial correlations, Rex systematically falsifies hypotheses against empirical telemetry across an interactive, multi-turn inquiry loop.

```mermaid
flowchart TD
    Alert["<b>Incoming Security Alert</b><br><i>(Login, Transaction, API Call)</i>"] --> Engine

    subgraph Engine ["<b>Rex Closed-Loop PoE Engine</b> <i>(Sub-Second SLA &lt; 1s)</i>"]
        direction TB
        subgraph Rounds ["Multi-Round Falsification &amp; Convergence"]
            direction LR
            subgraph R1 ["Round 1: Explore (K=3)"]
                direction TB
                A1["4D PoE Attention Bias<br><i>(Mask Dead H)</i>"] --> T1["ToolMLPHead<br><i>Network, Device, Behavior</i>"]
            end

            subgraph R2 ["Round 2: Focus (K=2)"]
                direction TB
                A2["4D PoE Attention Bias<br><i>(Mask Dead H)</i>"] --> T2["ToolMLPHead<br><i>Graph Telemetry</i>"]
            end

            subgraph R3 ["Round 3: Conclude (K=1)"]
                direction TB
                A3["4D PoE Attention Bias<br><i>(Mask Dead H)</i>"] --> V3["SharedVerdictHead<br><i>Calibrated Probabilities</i>"]
            end

            R1 -->|"New Evidence"| R2
            R2 -->|"Converged State"| R3
        end
    end

    T1 -->|"Outbound MCP"| V1["Client VPC Tools<br><i>(PII-Scrubbed)</i>"]
    T2 -->|"Outbound MCP"| V2["Client VPC Tools<br><i>(PII-Scrubbed)</i>"]
    V3 --> Final["<b>Final Disposition &amp; Runbook Dispatch</b><br><i>(Block / Challenge / Allow)</i>"]
```

---

## Key Architectural Pillars

### 1. Karl Popper’s Process of Elimination (PoE)

Scientific epistemology dictates that an empirical theory cannot be proven conclusively—it can only be rigorously falsified. Rex adopts **Monotonic Popperian Falsification**:

- Given necessary physical conditions for each attack vector, telemetry either satisfies or refutes the hypothesis.
- When an empirical metric violates a necessary bound (e.g., biological keystroke entropy refuting linear script execution), the corresponding hypothesis mask is permanently toggled to $M_j = 0$.
- Falsification is irreversible within an incident lifecycle, guaranteeing convergence and preventing oscillating decisions.

### 2. MECE Hypothesis Taxonomy

Rex replaces thousands of ad-hoc classification labels with 4 canonical, mutually exclusive, and collectively exhaustive root-cause hypotheses:

```mermaid
flowchart LR
    Incident["<b>Incident Telemetry</b>"] --> MECE["<b>MECE Hypothesis Space</b><br><i>(Mutually Exclusive, Collectively Exhaustive)</i>"]

    MECE --> H1["<b>H_botnet</b><br>Automated Botnets<br><i>Refuted by: Biological entropy &ge; 0.85</i>"]
    MECE --> H2["<b>H_sybil</b><br>Coordinated Syndicates<br><i>Refuted by: Unique canvas cluster size = 1</i>"]
    MECE --> H3["<b>H_shared</b><br>Legitimate Shared Egress<br><i>Refuted by: Datacenter proxy ASN</i>"]
    MECE --> H4["<b>H_novel</b><br>Zero-Day / Novel Abuse<br><i>Absorbent residual state if H1..H3 refuted</i>"]
```

| Hypothesis ID | Designation                     | Physical Failure Mode                                                                     | Representative Necessary Falsification Criterion                                                                        |
| ------------- | ------------------------------- | ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `H_botnet`    | **Automated Botnet Execution**  | Headless browsers, credential stuffing tools, automated emulation scripts.                | High keystroke delay variance ($\ge 120\text{ ms}$) or organic cursor curvature ($\text{entropy} \ge 0.85$).            |
| `H_sybil`     | **Coordinated Human Syndicate** | Device farms, fraud rings, multi-accounting, account takeover operations.                 | Singular unique canvas fingerprint cluster ($\text{size} = 1$) or zero shared financial instrument overlap.             |
| `H_shared`    | **Legitimate Shared Egress**    | Benign end-users sharing infrastructure (Carrier-Grade NAT, corporate VPN, campus Wi-Fi). | Commercial datacenter hosting ASN confirms datacenter proxy egress, refuting residential mobile CGNAT.                  |
| `H_novel`     | **Zero-Day / Novel Abuse**      | Uncataloged attack surface, protocol-level zero-days, unclassified exploitation.          | _Residual absorbant state:_ Activated when all known physical models ($H_{botnet}, H_{sybil}, H_{shared}$) are refuted. |

### 3. Dual-Head Multi-Task Inference

In each investigation turn, Rex executes a single, non-autoregressive forward pass delivering two coupled outputs:

1. **`ToolMLPHead`**: Evaluates the global incident state $[CLS]$ against a dynamic budget $K_t$, predicting the optimal set of semantic tool slots to query next.
2. **`SharedVerdictHead`**: Evaluates masked hypothesis positions $[MASK]_j$ to produce calibrated posterior probabilities across surviving branches.

### 4. Enterprise Security: Zero Inbound Ports

Rex connects to customer enterprise infrastructure exclusively via **Outbound-only Model Context Protocol (MCP)** or secure WebSockets. Enterprise telemetry remains inside the customer's private VPC; Rex dispatches tool queries outbound, receives sanitized responses with automated PII scrubbing (RFC 5322 emails, JWTs, bearer tokens, and payment card numbers stripped), eliminating network perimeter vulnerabilities.

---

## Neural Architecture & Dual-Head Projections

Rex builds upon an optimized Transformer Encoder foundation (such as ModernBERT with rotary embeddings and flash attention), augmented with specialized projection modules and attention interventions.

```mermaid
flowchart TD
    subgraph Seq ["<b>Input Sequence Layout</b>"]
        direction LR
        S_CLS["[CLS]"] --- S_INST["&lt;TASK_RISK_INVESTIGATION&gt; inst [SEP]"] --- S_M0["[MASK]_0 H_botnet"] --- S_M1["[MASK]_1 H_sybil"] --- S_M2["[MASK]_2 H_shared"] --- S_M3["[MASK]_3 H_novel [SEP]"] --- S_STATE["state_t [SEP]"]
    end

    S_CLS --> H_CLS["h_[CLS]"]
    S_M0 --> H_M0["h_[MASK]_0"]
    S_M1 --> H_M1["h_[MASK]_1"]
    S_M2 --> H_M2["h_[MASK]_2"]
    S_M3 --> H_M3["h_[MASK]_3"]

    H_CLS --> ToolHead["<b>ToolMLPHead</b><br>Linear(D, D) &rarr; LayerNorm<br>&rarr; GELU &rarr; Linear(D, 4)"]

    H_M0 --> VerdictHead["<b>SharedVerdictHead</b><br>Linear(D, 1)<br><i>(Shared across all 4 hypothesis masks)</i>"]
    H_M1 --> VerdictHead
    H_M2 --> VerdictHead
    H_M3 --> VerdictHead

    ToolHead --> ToolLogits["Tool Slot Logits<br><i>(Network, Device, Behavior, Graph)</i>"]
    VerdictHead --> VerdictLogits["Verdict Logits<br>[z_botnet, z_sybil, z_shared, z_novel]"]

    PoEMask["PoE Mask Vector M_j<br><i>(Refuted = 0)</i>"] --> LogitMask["<b>Hard Logit Masking</b><br>z_j = -10000.0 if M_j == 0"]
    VerdictLogits --> LogitMask
    LogitMask --> SoftmaxDist["<b>Softmax Distribution</b><br>P(H_j) with Novel Residual Mass"]

    SoftmaxDist --> InverseTool["<b>Inverse Tool Masking</b><br>Inhibit tools for refuted hypotheses"]
    ToolLogits --> InverseTool

    PoEMask -.-> AttentionBias["<b>4D Additive PoE Attention Bias [B, 1, S, S]</b><br>Zeroes cross-attention between refuted H_j and state_t"]
    AttentionBias -.-> S_STATE
```

### 1. Invariant Prefix & Sequence Formatting

To guarantee that positional encodings and token alignments remain mathematically stable across multi-turn inquiries, Rex enforces an **Invariant Prefix**:

- **Prefix Structure**: `[CLS] <type_token> instructions [SEP] [MASK]_0 H_0 [MASK]_1 H_1 [MASK]_2 H_2 [MASK]_3 H_3 [SEP]`
- **Compact Telemetry**: Raw JSON noise is serialized into compact key-value tokens (`k=v`), yielding a **~50% token reduction**.
- **Strict Left-Truncation**: When cumulative state telemetry exceeds sequence budget (`max_len = 512`), older telemetry tokens are discarded from the left, ensuring that fresh observations and the invariant prefix are strictly preserved.

### 2. Dual-Head Specifications

#### ToolMLPHead

Projects the global incident representation $h_{[CLS]} \in \mathbb{R}^D$ to scores for the 4 canonical semantic tool slots:
$$\text{Logits}_{\text{tool}} = W_2 \cdot \text{GELU}\left(\text{LayerNorm}(W_1 h_{[CLS]} + b_1)\right) + b_2 \in \mathbb{R}^4$$
Tool selection is constrained by:

- Dynamic round budget $K_t \in \{3, 2, 1\}$.
- Activation floor threshold $\tau_{\text{floor}} \ge 0.35$.
- Relative margin filter $\tau_{\text{margin}} \ge \max(\text{scores}) - 0.15$.
- **Inverse Tool Masking**: Tools associated exclusively with refuted hypotheses are mathematically inhibited.

#### SharedVerdictHead

To prevent head drift and parameter imbalance, a single shared projection matrix $W_v \in \mathbb{R}^{1 \times D}$ scores all hypothesis mask embeddings $h_{[MASK]_j}$:
$$z_j = W_v h_{[MASK]_j} + b_v \quad \forall j \in \{0, 1, 2, 3\}$$

### 3. Dual PoE Elimination Control Mechanisms

1. **4D Additive PoE Attention Bias Tensor ($[B, 1, S, S]$)**:
   Intervenes directly within the Transformer self-attention kernels. When hypothesis $H_j$ is eliminated ($M_j = 0$), all cross-attention weights between token positions of $H_j$ and the $[CLS]$ or telemetry tokens $state_t$ are set to $-10000.0$:
   $$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{Bias}_{\text{PoE}}\right) V$$
   This prevents eliminated hypotheses from distorting downstream hidden representations.

2. **Hard Logit Masking & Novel Residual Redistribution**:
   Logits of refuted hypotheses are hard-clamped prior to normalization:
   $$\tilde{z}_j = \begin{cases} z_j & \text{if } M_j = 1 \\ -10000.0 & \text{if } M_j = 0 \end{cases}$$
   $$P(H_j) = \frac{\exp(\tilde{z}_j)}{\sum_{k} \exp(\tilde{z}_k)}$$
   _Mathematical Guarantee:_ If all three concrete hypotheses are refuted ($M = [0, 0, 0, 1]$), the entire probability mass concentrates into $H_{\text{novel}}$:
   $$\lim_{\tilde{z}_{0,1,2} \to -\infty} P(H_{\text{novel}}) = 1.0$$

---

## Multi-Turn Closed-Loop Investigation Cycle

Rex operates as a budget-constrained funnel across discrete turns $t \in \{1, 2, 3\}$:

```mermaid
flowchart TD
    A([Incoming Alert Ingestion]) --> B[PII Scrubbing & Context Compaction]
    B --> C[Static Rule Pre-Forward Sync\nEvaluate necessary conditions]
    C --> D{Any H refuted?}
    D -- Yes --> E[Update PoE Mask M_j = 0]
    D -- No --> F[Maintain Current Mask]
    E --> G[Rex Multi-Task Forward Pass\nApply 4D PoE Attention Bias]
    F --> G
    G --> H[ToolMLPHead & Inverse Tool Masking]
    H --> I{Termination Criteria Met?\nConf >= 0.95 | Single Survivor | t = 3}
    I -- Yes --> J([Final Verdict & Action Dispatch])
    I -- No --> K[Dispatch Outbound MCP Tools\nFunnel Budget: K_1=3 -> K_2=2 -> K_3=1]
    K --> L[Sanitize New Telemetry]
    L --> M[Post-Tool Monotonic Falsification]
    M --> C
```

### Turn Lifecycle Steps

1. **Alert Ingestion & PII Sanitization**: Raw alert data is parsed, redacted, and formatted into compact telemetry notation.
2. **Static PoE Rule Synchronization**: Before running neural inference, active telemetry is checked against deterministic falsification conditions. Violations immediately eliminate candidate hypotheses ($M_j = 0$).
3. **Neural Forward Pass**: Transformer encoder processes the sequence with the 4D PoE attention bias tensor.
4. **Tool Slot Selection & Inverse Masking**: `ToolMLPHead` proposes tool queries within round budget $K_t$, automatically pruning tools belonging to refuted branches.
5. **Outbound Tool Execution**: Outbound MCP queries retrieve targeted observations from VPC telemetry stores.
6. **Asymmetric Early-Exit Check**: The inquiry terminates immediately if:
   - High confidence convergence: $\max P(H_j) \ge 0.95$.
   - Single surviving hypothesis: $\sum M_j = 1$.
   - Evidence sufficiency reached ($S \ge \tau_S$).
     _Result:_ Reduces average investigation rounds to **1.6–1.8 turns**, saving **~40% in telemetry invocation costs**.

---

## Semantic Tool Slots & Enterprise Outbound MCP

Rex interfaces with enterprise environments through 4 canonical Semantic Tool Slots, insulating neural planning from private schema changes:

| Semantic Slot   | Telemetry Capabilities                                                                                                          | Falsification Target                                              |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `SLOT_NETWORK`  | BGP ASN classification, carrier CGNAT indicators, residential proxy egress, TLS JA3/JA4 fingerprinting, TCP round-trip latency. | Falsifies $H_{botnet}$ (if benign ISP); confirms $H_{shared}$.    |
| `SLOT_DEVICE`   | WebGL/Canvas entropy, anti-detect browser footprints, headless Webdriver flags, hardware virtualization artifacts.              | Falsifies $H_{shared}$; confirms $H_{botnet}$ or $H_{sybil}$.     |
| `SLOT_BEHAVIOR` | Micro-keystroke cadence jitter, mouse trajectory curvature entropy, touch point area dynamics, request rate velocity.           | Falsifies $H_{botnet}$ (biological jitter proves human operator). |
| `SLOT_GRAPH`    | Identity cluster traversal, shared payment instruments, device tree overlap, referral graph density.                            | Falsifies $H_{shared}$; confirms $H_{sybil}$ syndicate.           |

### Outbound MCP Protocol Flow

```mermaid
flowchart LR
    subgraph VPC ["<b>Customer Private VPC (Zero Inbound Ports)</b>"]
        direction TB
        DB[("Internal SIEM / DBs /<br>Device Fingerprint Store")]
        SDK["<b>Rex Client SDK Agent</b><br><i>(Runs inside private perimeter)</i>"]
        Sanitizer["<b>Automated PII Redaction</b><br><i>(Scrubs RFC 5322, JWT, PAN)</i>"]
        DB <--> SDK
        SDK --> Sanitizer
    end

    subgraph Rex ["<b>Rex Engine (Isolated Controller)</b>"]
        direction TB
        Gateway["<b>MCP Dispatch Gateway</b><br><i>(WebSocket / JSON-RPC 2.0 Receiver)</i>"]
        Controller["<b>Investigation Controller</b><br><i>(Multi-Turn PoE Inference)</i>"]
        Gateway <--> Controller
    end

    Gateway -->|"<b>1. Outbound Query</b><br>tools/call (slots: network, device)"| SDK
    Sanitizer -->|"<b>2. Sanitized Telemetry</b><br>PII-Free Tokens (k=v)"| Gateway
```

---

## Action Dispatch & Incident Remediation

Once an investigation concludes, Rex dispatches structured remediation payloads:

```mermaid
flowchart TD
    Verdict{"<b>Concluded Verdict</b>"}

    Verdict -->|"H_botnet"| BotnetAction["<b>Botnet Automated Runbook</b><br>• Dynamic PoW challenge<br>• Null-route ASN / residential proxies<br>• Edge headless browser termination"]
    Verdict -->|"H_sybil"| SybilAction["<b>Sybil Syndicate Runbook</b><br>• Freeze pending transactions &amp; payouts<br>• Quarantine canvas fingerprint clusters<br>• Revoke promotional vouchers &amp; sessions"]
    Verdict -->|"H_shared"| SharedAction["<b>Shared Egress Runbook</b><br>• Whitelist mobile carrier CGNAT pools<br>• Shift limits to Device/Account tier<br>• Zero friction for legitimate users"]
    Verdict -->|"H_novel"| NovelAction["<b>Novel Attack Runbook</b><br>• Apply shadow friction &amp; covert PoW<br>• Compile SecOps cryptographic dossier<br>• Vectorized WAF backtest (&lt;1ms, FPR &le; 0.01%)"]
```

### 1. Automated Runbooks (Known Root Causes)

- **`H_botnet`**: Activate dynamic client-side Proof-of-Work (PoW) challenges; null-route autonomous system / residential proxy CIDRs; enforce headless browser termination at the CDN edge.
- **`H_sybil`**: Freeze pending transactions and payout methods; quarantine matched device canvas clusters; revoke promotional vouchers and terminate linked sessions.
- **`H_shared`**: Whitelist mobile carrier CGNAT IP ranges; transition security limits from IP-level to Device/Account-level; downgrade alert severity without disrupting user experience.

### 2. Novel Attack Handling & Vectorized WAF Backtesting (`H_novel`)

When zero-day exploitation is confirmed:

1. **Shadow Friction**: Introduces synthetic latency and covert proof-of-work to slow down automated attackers without revealing detection.
2. **SecOps Audit Dossier**: Assembles a complete cryptographic audit trace of all telemetry, falsified branches, and neural representations for Tier-3 SOC review.
3. **Vectorized WAF Backtesting**:
   - Automatically synthesizes candidate WAF mitigation rules.
   - Leverages `ColumnarTrafficCache` (NumPy vectorized columnar arrays) to backtest rules against **100,000 historical enterprise traffic logs in < 1.0 ms**.
   - If False Positive Rate satisfies strict production criteria ($\text{FPR} \le 0.01\%$), rule is automatically staged for WAF deployment.

---

## Multi-Task Training & Loss Dynamics

Rex is trained end-to-end via the joint multi-task loss formulation `RexMultiTaskLoss`:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{tool}} \mathcal{L}_{\text{tool}} + \lambda_{\text{verdict}} \mathcal{L}_{\text{verdict}} + \lambda_{\text{ortho}} \mathcal{L}_{\text{ortho}}$$

### 1. Tool Asymmetric Focal Loss ($\mathcal{L}_{\text{tool}}$)

Security tool invocations are sparse. To address severe negative class imbalance without suppressing critical alerts, we apply asymmetric focal penalties:
$$\mathcal{L}_{\text{tool}} = - \sum_{k=1}^{4} \left[ y_k (1 - p_k)^{\gamma_{\text{pos}}} \log(p_k) + (1 - y_k) p_k^{\gamma_{\text{neg}}} \log(1 - p_k) \right]$$
with $\gamma_{\text{neg}} = 2.0$ and $\gamma_{\text{pos}} = 0.0$, initialized with prior bias $b_k = \text{logit}(\pi_k)$.

### 2. Cost-Sensitive PoE Masked Cross-Entropy ($\mathcal{L}_{\text{verdict}}$)

Penalizes false accusations on legitimate shared users ($H_{\text{shared}} \to H_{\text{botnet}}$) significantly more heavily than delayed detection, computed strictly over unfalsified hypotheses:
$$\mathcal{L}_{\text{verdict}} = - \sum_{j \in \{j \mid M_j = 1\}} c_j \cdot y_j \log \left( \frac{\exp(\tilde{z}_j)}{\sum_{k: M_k = 1} \exp(\tilde{z}_k)} \right)$$

### 3. Tool Orthogonal Loss ($\mathcal{L}_{\text{ortho}}$)

Enforces geometric independence between the projection weights $W_{\text{tool}}$ of the 4 semantic slots, preventing redundant feature representation:
$$\mathcal{L}_{\text{ortho}} = \left\| W_{\text{tool}} W_{\text{tool}}^T - I \right\|_F^2$$

---

## Operational SLAs & Performance Benchmarks

Measured on standard production hardware (1x NVIDIA T4 GPU / Apple Silicon MPS / Modern 16-Core CPU):

| Operational Metric               | Target SLA         | Measured Benchmark                             | Business / Security Impact                                                 |
| -------------------------------- | ------------------ | ---------------------------------------------- | -------------------------------------------------------------------------- |
| **End-to-End Investigation SLA** | $< 1000\text{ ms}$ | **$215\text{ ms} - 450\text{ ms}$** (3 rounds) | Sub-second decisioning during user login / checkout without friction.      |
| **Single Forward Pass Latency**  | $< 40\text{ ms}$   | **$32.8\text{ ms}$** (T4 GPU)                  | Instantaneous System 1 inference for high-volume streams.                  |
| **Vectorized WAF Backtesting**   | $< 5.0\text{ ms}$  | **$< 1.0\text{ ms}$** (100k events)            | Real-time validation of candidate firewall rules against false positives.  |
| **Benign False Positive Rate**   | $\le 0.01\%$       | **$\le 0.008\%$** ($< 1$ in $12,500$)          | Eliminates disruption for legitimate users on cellular CGNAT / Wi-Fi.      |
| **Telemetry Cost Reduction**     | $> 30\%$           | **$41.2\%$** savings                           | Asymmetric early-exit resolves inquiries in $1.6$ rounds on average.       |
| **Inbound Network Exposure**     | **0 ports**        | **0 open ports**                               | Pure outbound WebSocket / MCP architecture; zero perimeter attack surface. |

---

## Installation

### Prerequisites

- Python 3.10, 3.11, 3.12, or 3.13
- PyTorch 2.0+ (CUDA, MPS, or CPU)

### Install via pip / uv

```bash
# Core package installation
pip install rex

# With optional extras:
pip install "rex[serve]"       # Production HTTP server (FastAPI + Uvicorn)
pip install "rex[mcp]"         # Model Context Protocol stdio/SSE server
pip install "rex[structured]"  # Pydantic schema validation helpers
pip install "rex[onnx]"        # ONNX Runtime inference & export
pip install "rex[fast]"        # High-performance GPU kernels via TileLang
```

Using [uv](https://docs.astral.sh/uv/):

```bash
uv add rex
```

### Install from Source

```bash
git clone https://github.com/RequinLabs/rex.git
cd rex
pip install -e ".[serve,mcp]"
```

---

## Quickstart

### 1. Autonomous Multi-Turn Investigation Loop

The `InvestigatorAgent` autonomously orchestrates multi-turn inquiries, queries tools through semantic slots, and applies Popperian pruning until convergence:

```python
from rex.events.investigation_loop import InvestigationLoopController
from rex.events.registry_loader import get_master_registry

# Load default MECE Master Registry (defining H_botnet, H_sybil, H_shared, H_novel)
registry = get_master_registry()

# Initialize investigation loop controller
controller = InvestigationLoopController(
    domain_type="TYPE_CREDENTIAL_STUFFING",
    max_rounds=3,
    confidence_threshold=0.95,
)

# Simulated suspicious alert telemetry
incoming_alert = {
    "network": {
        "ip": "198.51.100.24",
        "asn": "AS15169",
        "is_datacenter": False,
        "is_cgnat": True,
    },
    "device": {
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "canvas_cluster_size": 1,
    },
    "behavior": {
        "keystroke_jitter_ms": 145.0,  # High jitter (> 120ms) refutes bot automation
        "mouse_entropy": 0.92,         # Organic curvature refutes scripts
    }
}

# Define your VPC tool dispatcher callback
def vpc_tool_dispatcher(requested_slots: list[str]) -> dict:
    telemetry = {}
    if "SLOT_GRAPH" in requested_slots:
        # Client queries internal account graph database
        telemetry["graph"] = {"linked_payout_overlap": 0, "cluster_size": 1}
    return telemetry

# Run the closed-loop investigation
session_result = controller.run_investigation(
    initial_alert=incoming_alert,
    tool_dispatcher=vpc_tool_dispatcher,
)

print(f"Session ID        : {session_result.session_id}")
print(f"Final Verdict     : {session_result.final_verdict}")       # H_shared
print(f"Confidence        : {session_result.final_confidence:.4f}")  # 0.96+
print(f"Rounds Taken      : {session_result.total_rounds}")        # 1 or 2 (Early Exit)
print(f"Falsified Branches: {session_result.falsified_trail}")
print(f"Runbook Action    : {session_result.action_payload['title']}")
```

### 2. High-Speed Single-Pass Decision Routing

For instant classification across text, emails, or structured JSON payloads without multi-turn loops:

```python
from rex import Router

# Preload checkpoints into memory for instant sub-35ms routing
router = Router(preload=True)

state = {
    "subject": "Urgent wire transfer request",
    "body": "Please wire $50,000 to the attached vendor account immediately before EOD.",
    "sender_domain_age_days": 2,
}

questions = {
    "threat_category": {
        "type": "choice",
        "instructions": "Classify incoming request risk category.",
        "criteria": {
            "bec_phishing": "Business email compromise, urgent wire transfers, CEO spoofing",
            "account_takeover": "Credentials, MFA bypass, password resets",
            "benign": "Standard operational correspondence"
        }
    },
    "action_urgency": {
        "type": "score",
        "instructions": "Evaluate risk severity level.",
        "criteria": ["low", "medium", "critical_block"]
    }
}

result = router.predict(state, questions)
print("Classification:", result["answers"]["threat_category"]["choice"])  # bec_phishing
print("Confidence    :", result["answers"]["threat_category"]["confidence"])
```

Passing an empty question dictionary to `agent.predict(state, {})` or `agent.system_one(state, {})` returns the standard response with `"answers": {}` and `"usage": {"input_tokens": 0, "output_tokens": 0}`. The state is not tokenized and no model forward pass runs.

#### Batch Processing & Lifecycle Hooks

Execute high-throughput batch predictions with custom hooks:

```python
results = router.predict_batch(requests)
```

* **Per-call hooks reach the batch path.** `Router.predict_batch` takes `hooks`, `on_predict_start`, `on_predict_end`, `hooks_raise` and `hooks_timeout`, matching `predict`; `Router.route_batch` takes `hooks`, `hooks_raise` and `hooks_timeout`, since it fires only `on_route` and has no predict events for the convenience callables to bind to (#909). The batched dispatch also composes installed hooks properly, so a `set_default_hooks` default no longer fires on `predict` and silently vanishes on `predict_batch`.

#### Long Context Document Scanning

For long documents exceeding standard context length, `agent.predict_long(state, questions)` windows the state:

- A state that already fits one window is passed straight to `system_one` (identical output, plus `usage["windows"] = 1`). Without an explicit `window`, that is any state the questions leave room for, since `system_one` reads it whole. The key is total: `1` single window, `N` scanned windows, `0` a hook answered the document before the model read any of it.
- Hooks wrap the inference that answers the state, so on a scanned document `on_predict_start` fires once with `ctx.states` holding the decoded windows, not the state you passed in (it was tokenized to produce them). A start hook may replace that list: the answers are aggregated over whatever reached inference, and `usage["windows"]` counts those states. What a rewritten scan costs is the attribution -- `answer["window"]` names a span of *your* document, so it is only reported when the scan reached inference unchanged. A hook that means to answer the document calls `ctx.skip(...)` instead: its result comes back with no `answer["window"]` and `usage["windows"]` at 0, because no window scored it.

A smaller `window` isolates a short deciding span better (it becomes a larger fraction of its window); the default (`max(64, max_len - head_max_len - 8)` -- 312 tokens on the English checkpoint at its 512/192 defaults, 760 on the other two at 1,024/256) favors context and throughput. Either way the window is capped at the room the questions leave for the state inside `max_len`, so a window is never re-truncated on the way to the model and `token_start`/`token_end` describe the span it read.

* **High-cardinality choice questions and token budgets:** Sequences split into an option prompt budget (`head_max_len`) and the remaining document/state budget. `head_max_len` is a cap, not the head length. The renderer fills the option prompt budget only as far as the question and its option descriptions need, and `build_sequence` then sizes the state from the head it actually built (`room = max_len - head_len - 1`). The real room therefore depends on the question, not on the cap:
  * `english` defaults to 512 context (`head_max_len = 192`). The shipped department question renders a **48-token** head, leaving **463 tokens for state**, not 320.
  * `multilingual` and `typed-decisions` default to 1,024 context (`head_max_len = 256`). The shipped question renders a **45-token** head, leaving **978 tokens for state**, not 768.

#### Calibration

Both checkpoints are over-confident as shipped. Refitting one temperature per (question type,
option count) on held-out data moves mean ECE **0.466 -> 0.081** (`rex`) and
**0.314 -> 0.106** (`rex-multilingual`). `rex-multilingual` ships with no fitted
temperatures at all, so fit them before relying on its probabilities.

### 3. Outbound MCP Tool Integration

Deploy the Rex MCP server for seamless integration with Model Context Protocol agents:

The tools return structured JSON. `laya_predict_batch` and `laya_route_batch` are the MCP form of `Router.predict_batch` / `route_batch`: one tool call takes an array of `{state, questions, model?, task?, lang?, lang_guess?, max_len?, head_max_len?}` requests, routes them first, groups them by checkpoint, and shares forward passes between requests with the same question schema, returning the answers in input order.

`laya_shortlist` is the MCP form of `predict_shortlist`: it shortlists a many-option choice question to its `k` most likely labels by embedding similarity, answers in one forward pass, and returns per-question shortlist metadata (kept labels, cosine scores, `k`, option count, and whether the question passed through unshortlisted).

```bash
# Launch the Rex MCP Server
rex-mcp-server --port 8080
```

Register tools natively in Python via the `ToolRegistry`:

```python
from rex.events.falsification import ToolRegistry, ToolSlot

registry = ToolRegistry()

@registry.register(ToolSlot.NETWORK)
def fetch_network_telemetry(ip: str) -> dict:
    # Query your internal IP reputation service
    return {
        "is_datacenter": False,
        "carrier_cgnat": True,
        "ja3_fingerprint_hash": "e7d705a32831b26160177c33a5168b18"
    }
```

---

## Production Deployment

### HTTP Inference Microservice (`rex-serve`)

Rex includes a production FastAPI microservice supporting single-shot decisions, batch processing, and live audit telemetry:

```bash
# Start server on 0.0.0.0:8000
rex-serve --host 0.0.0.0 --port 8000 --device cuda
```

Query the API via curl:

```bash
curl -X POST http://localhost:8000/v1/investigate \
  -H "Content-Type: application/json" \
  -d '{
    "domain_type": "TYPE_CREDENTIAL_STUFFING",
    "telemetry": {
      "network": {"ip": "198.51.100.12", "is_datacenter": true},
      "behavior": {"mouse_entropy": 0.12}
    }
  }'
```

### Command-Line Interface (`rex`)

Inspect alerts, run batch evaluations, and verify configurations from your terminal:

```bash
# Quick incident evaluation
rex "High frequency login failures from residential IP cluster" --preset triage

# Process a batch of alerts in one forward pass
rex --batch alerts.jsonl --predict --json > results.jsonl

# Inspect model configurations and tool slots
rex-evals --model-path ./checkpoints/rex-base
```

---

## Model Training & Kaggle Workflow

Rex provides full training and fine-tuning pipelines for custom RBA, fraud, and security telemetry datasets.

```bash
# Train Rex with the multi-task PoE objective
rex-train \
  --config configs/train_rba.yaml \
  --data-path data/rba_telemetry.jsonl \
  --output-dir checkpoints/rex-rba \
  --epochs 5 \
  --batch-size 32 \
  --lr 3e-5
```

### Reference Kaggle Notebook

A complete, self-contained end-to-end training notebook for the Risk-Based Authentication (RBA) dataset is located at:
[`notebooks/rex_rba_kaggle_training.ipynb`](https://github.com/RequinLabs/rex/blob/main/notebooks/rex_rba_kaggle_training.ipynb)

The notebook guides you through:

1. Environment setup and checkpoint hydration.
2. Synthesizing multi-round investigation episodes from static fraud datasets.
3. Training with `RexMultiTaskLoss` (Tool Asymmetric Focal Loss + Cost-Sensitive PoE Masked Cross-Entropy + Orthogonal Loss).
4. Evaluating convergence rates, temperature calibration, and WAF candidate rule generation.

---

## Project Structure

```
rex/
├── rex/
│   ├── events/                 # Popperian PoE & MECE Investigation Engine
│   │   ├── backbone.py         # Invariant prefix formatting, ToolMLPHead, SharedVerdictHead
│   │   ├── falsification.py    # Monotonic falsification rules & slot dependency matrix
│   │   ├── investigation_loop.py # Multi-turn closed-loop controller & budget pruner
│   │   ├── mece.py             # Canonical MECE hypothesis definitions
│   │   ├── model_modifier.py   # Embedding expansion & semantic warm-start
│   │   ├── registry_loader.py  # Master registry configuration loader & validator
│   │   └── configs/            # Production YAML/JSON registries (RBA, API Anomaly, etc.)
│   ├── mcp/                    # Model Context Protocol tools & outbound server
│   ├── orch/                   # Persistent orchestration, SQLite stores & session state
│   ├── backends/               # Execution backends (PyTorch Eager, TileLang GPU fast path)
│   ├── agent.py                # Base decision agent & hook registry
│   ├── rex_agent.py            # InvestigatorAgent procedural loop orchestrator
│   ├── rex_model.py            # Unified RexDecisionModel (arch="rex2")
│   ├── rex_heads.py            # Decision heads (Sufficiency, Risk, Calibrators)
│   ├── losses.py               # RexMultiTaskLoss, ToolAsymmetricFocalLoss, OrthogonalLoss
│   ├── rex_train.py            # Multi-task training engine & evaluation loops
│   ├── router.py               # Multi-checkpoint language & task router
│   └── serve.py                # FastAPI HTTP microservice
├── notebooks/                  # Training & evaluation notebooks (Kaggle RBA, Credit Card)
├── tests/                      # Automated unit, integration, and API contract test suite
├── pyproject.toml              # Build specifications & dependency extras
└── README.md                   # System documentation & architectural reference
```

---

## Try it locally: web GUI + JSON API

`examples/server.py` is a self-contained FastAPI app for testing without writing any code: a two-pane playground plus a plain JSON API (`/predict`, `/predict/batch`) for scripting against.

```bash
pip install "rex[serve]"
python examples/server.py               # http://127.0.0.1:8000
```

A request body may carry `model` to pin a checkpoint instead of letting the router choose, and it takes exactly the spellings `rex --model` takes: names, aliases and casing all resolve through `rex.router`. `GET /models` lists the checkpoints and the aliases alongside them.

`/predict/batch` accepts two optional body fields that control the shape of the forward passes without changing any answer: `batch_size` (states per pass; omit it and the whole batch is one pass) and `sort_by_length` (group similarly sized states so each pass pads to a shorter maximum).

---

## Contributing & Development Guidelines

We welcome community contributions, custom tool slot integrations, and new falsification domain registries!

Before submitting a Pull Request, please ensure all CI checks pass:

```bash
# Syntax and bytecode compilation validation
python -m compileall -q rex/ tests/

# Run the test suite
pytest tests/
```

- **Commit Conventions**: Use conventional commits matching repository history (`feat(events):`, `fix(poe):`, `perf(model):`, `docs(readme):`, `test(loop):`).
- **No Inbound Dependencies**: Rex core must operate strictly locally or in-process; dependencies on external hosted APIs will not be merged.
- **Contract Verification**: Modifications to public APIs must update corresponding tests under `tests/test_hooks_api.py`.

---

## License

Rex is licensed under the [Apache License, Version 2.0](https://github.com/RequinLabs/rex/blob/main/LICENSE).  
Copyright © 2026 RequinLabs / Rex Contributors.
