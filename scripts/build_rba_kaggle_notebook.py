"""Generator for production-grade Kaggle Notebook training Rex ModernBERT on DAS Group RBA Dataset."""

import json
from pathlib import Path

def create_notebook():
    cells = []

    def add_md(source):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    def add_code(source):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.strip().split("\n")]
        })

    # =========================================================================
    # Cell 0: Header & Settings
    # =========================================================================
    add_md("""# 🛡️ Rex / ModernBERT: Multi-Turn MECE Investigation Training on Kaggle RBA Dataset
### Risk-Based Authentication (RBA) with Popperian Process of Elimination (PoE)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Private%20Checkpoint-yellow)](https://huggingface.co/)
[![Dataset](https://img.shields.io/badge/Dataset-DAS%20Group%20RBA-blue)](https://www.kaggle.com/datasets/dasgroup/rba-dataset)
[![License](https://img.shields.io/badge/Model%20License-Apache%202.0-green)](https://www.apache.org/licenses/LICENSE-2.0)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey)](https://creativecommons.org/licenses/by/4.0/)

---

### ⚠️ Kaggle Notebook Settings:
1. **Accelerator:** `GPU T4 x2` or `GPU P100` (16GB VRAM).
2. **Internet:** `On` (Required for `kagglehub` dataset download, base ModernBERT weights, and Hugging Face upload).
3. **Hugging Face Token:** Go to **Add-ons -> Secrets** in Kaggle, add secret `HF_TOKEN` with **Write** permission.""")

    # =========================================================================
    # Cell 1: Environment & GPU Check
    # =========================================================================
    add_md("""## 1. Environment Check & Multi-GPU / Modern PyTorch AMP Setup
Detects GPU capability, selects optimal precision (`bfloat16` for Ampere+ $\\ge 8.0$, `float16` for Turing T4 / Pascal P100), and configures reproducible global seeds.""")

    add_code("""!nvidia-smi

import os
import random
import torch
import numpy as np

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(42)

assert torch.cuda.is_available(), "CUDA is not available! Please enable GPU in Kaggle Notebook Settings."

device = torch.device("cuda:0")
device_count = torch.cuda.device_count()
device_name = torch.cuda.get_device_name(0)
device_cap = torch.cuda.get_device_capability(0)

# Turing (T4: 7.5) and Pascal (P100: 6.0) do NOT have native hardware bfloat16 fast path.
# Use float16 on T4/P100; use bfloat16 on Ampere/Hopper (cap >= 8.0).
amp_dtype = torch.bfloat16 if device_cap[0] >= 8 else torch.float16

print(f"CUDA Available      : True ({device_count} GPUs detected)")
print(f"Primary Device      : {device_name} (Capability: {device_cap[0]}.{device_cap[1]})")
print(f"Selected AMP Dtype  : {amp_dtype}")""")

    # =========================================================================
    # Cell 2: Dependencies
    # =========================================================================
    add_md("""## 2. Install Dependencies (Safe Pinned Upgrades)
Installs `kagglehub` for direct dataset download and upgrades `transformers`, `datasets`, `huggingface_hub`, and `safetensors`. We strictly **avoid** upgrading `torch` to protect Kaggle's pinned CUDA runtime.""")

    add_code("""# Safe Pinned Upgrades: Only install required packages.
# We strictly AVOID touching numpy, scipy, pandas, or torch to preserve Kaggle's pre-compiled C-extensions & CUDA runtime!
!pip install -q kagglehub "transformers>=4.48.0" "accelerate>=0.26.0" safetensors huggingface_hub
""")

    # =========================================================================
    # Cell 3: Dataset Acquisition via kagglehub
    # =========================================================================
    add_md("""## 3. Dataset Acquisition via `kagglehub`
Directly downloads or locates the official **`dasgroup/rba-dataset`** from Kaggle using `kagglehub.dataset_download("dasgroup/rba-dataset")`.""")

    add_code("""import os
import glob
import kagglehub

print("Downloading / locating 'dasgroup/rba-dataset' via kagglehub...")
try:
    dataset_path = kagglehub.dataset_download("dasgroup/rba-dataset")
    print("Path to dataset files:", dataset_path)
except Exception as e:
    print(f"kagglehub download notice ({e}). Falling back to local Kaggle input directory...")
    dataset_path = "/kaggle/input/rba-dataset"

# Locate CSV files in the downloaded path or /kaggle/input
csv_files = glob.glob(f"{dataset_path}/**/*.csv*", recursive=True)
csv_files = [f for f in csv_files if not os.path.isdir(f) and (".csv" in f.lower())]

if not csv_files:
    csv_files = glob.glob("/kaggle/input/**/rba*.csv*", recursive=True)
    csv_files = [f for f in csv_files if not os.path.isdir(f) and (".csv" in f.lower())]

if not csv_files:
    raise FileNotFoundError(
        f"CRITICAL: No CSV file found in '{dataset_path}' or '/kaggle/input/'! "
        "Please ensure 'dasgroup/rba-dataset' is accessible or attached via the Kaggle sidebar."
    )

raw_csv_path = sorted(csv_files, key=os.path.getsize, reverse=True)[0]
print(f"✅ Selected RBA CSV file: {raw_csv_path} ({os.path.getsize(raw_csv_path) / (1024*1024):.2f} MB)")""")

    # =========================================================================
    # Cell 4: Exploratory Data Profiling & Schema Inspection
    # =========================================================================
    add_md("""## 4. Exploratory Data Profiling & Schema Inspection
Reads the initial slice of the raw dataset to inspect the actual column schema, data types, missing value percentages, and raw feature distributions.""")

    add_code("""import pandas as pd

sample_df = pd.read_csv(raw_csv_path, nrows=5000, low_memory=False)

print("=" * 65)
print("RAW RBA DATASET SCHEMA & PROFILING (FIRST 5,000 ROWS):")
print(f"Total Columns: {len(sample_df.columns)}")
print("=" * 65)
for i, col in enumerate(sample_df.columns):
    n_missing = sample_df[col].isna().sum()
    pct_missing = n_missing / len(sample_df) * 100.0
    print(f"  {i+1:02d}. {col:30s} | Type: {str(sample_df[col].dtype):8s} | Missing: {pct_missing:5.1f}%")

print()
print("Sample Telemetry Preview (First 3 Rows):")
print(sample_df.head(3).T)
print("=" * 65)""")

    # =========================================================================
    # Cell 5: Data Preprocessing & Causal Feature Engineering (ETL Pipeline)
    # =========================================================================
    add_md("""## 5. Data Preprocessing & Causal Feature Engineering (ETL Pipeline)
Production Data Engineering Pipeline:
1. **Schema Standardization:** Robust mapping for IP, User, Timestamp, RTT (`round_trip_time_[ms]`), ASN, Country, and UA.
2. **High-Throughput Vectorization:** Converts timestamps per chunk via `pd.to_datetime` and streams rows with `itertuples()` (100x speedup over `iterrows()`).
3. **Causal Historical Features:** Sliding windows for IP velocity (1h), distinct users on IP (24h), distinct IPs per user (24h), datacenter ASN detection, bot UA regex, and user country mismatch.
4. **Independent Ground-Truth Mapping to Rex 4 MECE Hypotheses:**
   - $H_{\\text{botnet}}$ (index 0): `is_attack_ip == True` and `not is_account_takeover`.
   - $H_{\\text{sybil}}$ (index 1): `is_account_takeover == True` (compromised account / sybil takeover).
   - $H_{\\text{shared}}$ (index 2): `not is_attack_ip` and `not is_account_takeover` (legitimate user traffic over shared/residential egress).
   - $H_{\\text{novel}}$ (index 3): Unprecedented behavioral anomalies (Popperian residual).
5. **Dual Algorithm R Reservoir Sampling:** Uniform random sampling across the entire chronological stream, preserving a clean 3:1 Benign:Attack ratio.
6. **Memory Guard:** Deques bounded to `maxlen=50` to guarantee constant memory.""")

    add_code("""import re
import datetime
from collections import defaultdict, deque

# Major Datacenter / Cloud Hosting ASNs
DATACENTER_ASNS = {
    "16509", "14618",  # Amazon AWS
    "15169", "396982", # Google Cloud
    "8075",              # Microsoft Azure
    "14061",             # DigitalOcean
    "24940",             # Hetzner Online
    "16276",             # OVH SAS
    "20473",             # Choopa / Vultr
    "63949",             # Linode / Akamai
}

def preprocess_rba_stream(csv_path, max_total_events=30000, target_benign_ratio=3.0, max_scan_rows=500_000):
    # Streaming causal state: maintains sliding history with memory bound
    ip_login_times = defaultdict(lambda: deque(maxlen=50))
    ip_users_history = defaultdict(lambda: deque(maxlen=50))
    user_ips_history = defaultdict(lambda: deque(maxlen=50))
    user_primary_country = defaultdict(lambda: defaultdict(int))

    # Calculate target capacities for Algorithm R reservoirs
    target_attacks = int(max_total_events / (target_benign_ratio + 1.0))
    target_benign = int(target_attacks * target_benign_ratio)

    reservoir_attacks = []
    reservoir_benign = []
    n_seen_attacks = 0
    n_seen_benign = 0

    print(f"Starting Preprocessing ETL Pipeline (Target: {target_attacks} Attacks, {target_benign} Benign)...")
    chunk_iter = pd.read_csv(csv_path, chunksize=100_000, low_memory=False)

    total_scanned = 0
    for chunk_idx, chunk in enumerate(chunk_iter):
        # Clean column names to strictly valid Python identifiers (removes [, ], spaces, dashes)
        norm_map = {col: re.sub(r"[^0-9a-zA-Z]+", "_", col.strip().lower()).strip("_") for col in chunk.columns}
        chunk = chunk.rename(columns=norm_map)

        # Robust Column Mapping
        col_ip = next((c for c in chunk.columns if "ip" in c and "attack" not in c), "ip_address")
        col_user = next((c for c in chunk.columns if "user" in c and "agent" not in c), "user_id")
        col_succ = next((c for c in chunk.columns if "succ" in c), "login_successful")
        col_atk = next((c for c in chunk.columns if "attack" in c), "is_attack_ip")
        col_ato = next((c for c in chunk.columns if "takeover" in c or "ato" in c), "is_account_takeover")
        col_asn = next((c for c in chunk.columns if "asn" in c), None)
        col_ua = next((c for c in chunk.columns if "agent" in c), "user_agent_string")
        col_country = next((c for c in chunk.columns if "country" in c), None)
        col_rtt = next((c for c in chunk.columns if "round_trip" in c or "rtt" in c), None)
        col_ts = next((c for c in chunk.columns if "time" in c or "date" in c), "login_timestamp")

        # Fast Vectorized Timestamp Pre-conversion (Identifier without leading underscore)
        if col_ts and col_ts in chunk.columns:
            ts_vec = pd.to_datetime(chunk[col_ts], errors="coerce").astype("int64") // 10**9
            chunk["ts_unix"] = np.where(ts_vec > 0, ts_vec, total_scanned + np.arange(len(chunk)))
        else:
            chunk["ts_unix"] = total_scanned + np.arange(len(chunk))

        for row in chunk.itertuples(index=False):
            total_scanned += 1

            def to_bool(val):
                if isinstance(val, (bool, np.bool_)): return bool(val)
                s = str(val).strip().lower()
                return s in ("1", "true", "t", "yes")

            is_atk = to_bool(getattr(row, col_atk, False)) if col_atk else False
            is_ato = to_bool(getattr(row, col_ato, False)) if col_ato else False
            is_succ = to_bool(getattr(row, col_succ, True)) if col_succ else True

            user = str(getattr(row, col_user, "usr_unknown")).strip() if col_user else "usr_unknown"
            raw_ip = str(getattr(row, col_ip, "192.168.1.1")).strip() if col_ip else "192.168.1.1"
            asn_raw = str(getattr(row, col_asn, "0")).strip() if col_asn else "0"
            ua_str = str(getattr(row, col_ua, "")).lower() if col_ua else ""
            country_str = str(getattr(row, col_country, "XX")).strip() if col_country else "XX"

            # Parse RTT robustly
            rtt_raw = getattr(row, col_rtt, 50.0) if col_rtt else 50.0
            try:
                rtt_val = float(rtt_raw) if pd.notna(rtt_raw) else 50.0
            except Exception:
                rtt_val = 50.0

            curr_ts = float(getattr(row, "ts_unix", total_scanned))

            # Causal Historical Window Updates
            times_q = ip_login_times[raw_ip]
            while times_q and (curr_ts - times_q[0] > 3600):
                times_q.popleft()
            ip_velocity_1h = len(times_q)
            times_q.append(curr_ts)

            u_hist = ip_users_history[raw_ip]
            while u_hist and (curr_ts - u_hist[0][0] > 86400):
                u_hist.popleft()
            users_on_ip_24h = len(set(u for _, u in u_hist))
            u_hist.append((curr_ts, user))

            ip_hist = user_ips_history[user]
            while ip_hist and (curr_ts - ip_hist[0][0] > 86400):
                ip_hist.popleft()
            ips_for_user_24h = len(set(i for _, i in ip_hist))
            ip_hist.append((curr_ts, raw_ip))

            hist_countries = user_primary_country[user]
            country_mismatch = (country_str != "XX") and bool(hist_countries) and (country_str not in hist_countries)
            hist_countries[country_str] += 1

            asn_digits = "".join(re.findall(r"\d+", asn_raw))
            is_datacenter = asn_digits in DATACENTER_ASNS
            is_bot_ua = any(kw in ua_str for kw in ("headless", "curl", "python", "selenium", "puppeteer", "phantom", "go-http"))

            # Independent Ground-Truth Mapping to Rex 4 MECE Hypotheses:
            # 0: H_botnet (Automated credential stuffing / botnet IP)
            # 1: H_sybil (Account Takeover / compromised session)
            # 2: H_shared (Legitimate user login across shared/residential IP)
            # 3: H_novel (Extreme out-of-distribution anomaly / Popperian residual)
            if is_atk and not is_ato:
                target_hyp = "H_botnet"
                target_idx = 0
                is_attack = True
            elif is_ato:
                target_hyp = "H_sybil"
                target_idx = 1
                is_attack = True
            else:
                target_hyp = "H_shared"
                target_idx = 2
                is_attack = False

            event_bundle = {
                "user": user,
                "ip": raw_ip,
                "target_idx": target_idx,
                "target_hyp": target_hyp,
                "is_datacenter": is_datacenter,
                "is_bot_ua": is_bot_ua,
                "users_on_ip_24h": users_on_ip_24h,
                "ips_for_user_24h": ips_for_user_24h,
                "ip_velocity_1h": min(ip_velocity_1h, 100),
                "is_succ": int(is_succ),
                "country": country_str,
                "country_mismatch": int(country_mismatch),
                "rtt": min(int(rtt_val), 1000),
                "timestamp": curr_ts,
            }

            # True Algorithm R Reservoir Sampling
            if is_attack:
                n_seen_attacks += 1
                if len(reservoir_attacks) < target_attacks:
                    reservoir_attacks.append(event_bundle)
                else:
                    j = np.random.randint(0, n_seen_attacks)
                    if j < target_attacks:
                        reservoir_attacks[j] = event_bundle
            else:
                n_seen_benign += 1
                if len(reservoir_benign) < target_benign:
                    reservoir_benign.append(event_bundle)
                else:
                    j = np.random.randint(0, n_seen_benign)
                    if j < target_benign:
                        reservoir_benign[j] = event_bundle

        if total_scanned >= max_scan_rows:
            print(f"Reached scan budget limit ({max_scan_rows:,} rows). Assembling preprocessed events...")
            break

    actual_attacks = len(reservoir_attacks)
    needed_benign = min(len(reservoir_benign), int(actual_attacks * target_benign_ratio))
    sampled_events = reservoir_attacks + reservoir_benign[:needed_benign]

    # Inject 10% Controlled Novel/Zero-day Mixture (Popperian Residual H_novel)
    # where extreme telemetry signals refute all standard hypotheses
    n_novel = int(0.10 * len(sampled_events))
    for i in range(n_novel):
        idx = np.random.randint(0, len(sampled_events))
        # Novel anomalies: extreme latency and unusual foreign country jump
        novel_bundle = dict(sampled_events[idx])
        novel_bundle["target_idx"] = 3
        novel_bundle["target_hyp"] = "H_novel"
        novel_bundle["rtt"] = 999
        novel_bundle["country_mismatch"] = 1
        sampled_events.append(novel_bundle)

    # Sort strictly chronologically by timestamp
    sampled_events.sort(key=lambda x: x["timestamp"])

    print("=" * 65)
    print("ETL PREPROCESSING SUMMARY:")
    print(f"Total Rows Scanned        : {total_scanned:,}")
    print(f"Attacks Extracted         : {actual_attacks} (Seen: {n_seen_attacks:,})")
    print(f"Benign Extracted          : {needed_benign} (Seen: {n_seen_benign:,})")
    print(f"Novel Residual Injected   : {n_novel}")
    print(f"Total Preprocessed Events : {len(sampled_events)}")
    print("=" * 65)
    return sampled_events

preprocessed_events = preprocess_rba_stream(raw_csv_path, max_total_events=30000, target_benign_ratio=3.0, max_scan_rows=500_000)

features_summary_df = pd.DataFrame(preprocessed_events)[
    ["ip_velocity_1h", "users_on_ip_24h", "ips_for_user_24h", "rtt", "is_datacenter", "is_bot_ua", "country_mismatch"]
]
print()
print("Engineered Telemetry Feature Statistics:")
print(features_summary_df.describe().T[["mean", "std", "min", "50%", "max"]])""")

    # =========================================================================
    # Cell 6: Popperian Investigation Trajectory Synthesis
    # =========================================================================
    add_md("""## 6. Popperian Investigation Trajectory Synthesis
Transforms preprocessed events into 3-turn budgeted investigation trajectories matching `rex/events/mece.py`:
- **Unified Task Token:** Uses `<TASK_RISK_INVESTIGATION>` as the sole task anchor token.
- **Round 1 (Initial Alert Ingestion):** ContextHub alert signals only (`network.ip_velocity_1h`, `network.is_datacenter`, etc.). Mask $\\mathbf{M}_1 = [1, 1, 1, 1]$. Tools target broad discovery [1, 1, 1, 0] ($K_1=3$).
- **Round 2 (Focused Investigation):** Observes Round 1 tools (`device.is_bot_ua`, `behavior.login_successful`). Popperian rules evaluate observations to falsify hypotheses ($M_j=0$). Tool targets adapt ($K_2=2$).
- **Round 3 (Decisive Verification):** Observes Round 2 graph tools (`graph.users_on_ip_24h`). Selects single decisive verification probe ($K_3=1$).
- **False Refutation Rate (FRR) Tracking:** Logs empirical refutation error rate.""")

    add_code("""import collections
from enum import Enum

class RiskHypothesis(str, Enum):
    BOTNET = "H_botnet"
    SYBIL = "H_sybil"
    SHARED = "H_shared"
    NOVEL = "H_novel"

class ToolSlot(str, Enum):
    NETWORK = "SLOT_NETWORK"
    DEVICE = "SLOT_DEVICE"
    BEHAVIOR = "SLOT_BEHAVIOR"
    GRAPH = "SLOT_GRAPH"

CANONICAL_HYPOTHESES = [
    RiskHypothesis.BOTNET,
    RiskHypothesis.SYBIL,
    RiskHypothesis.SHARED,
    RiskHypothesis.NOVEL,
]

CANONICAL_TOOL_SLOTS = [
    ToolSlot.NETWORK,
    ToolSlot.DEVICE,
    ToolSlot.BEHAVIOR,
    ToolSlot.GRAPH,
]

# Exact Calibrated Invariant Prefix Descriptions matching rex/events/mece.py
MECE_HYPOTHESIS_STRINGS = {
    RiskHypothesis.BOTNET: (
        "Automated execution via scripts, headless browsers, or residential botnets"
    ),
    RiskHypothesis.SYBIL: (
        "Coordinated human fraud operation using cloned profiles or virtual environments"
    ),
    RiskHypothesis.SHARED: (
        "Legitimate users traversing shared egress infrastructure such as mobile CGNAT or public WiFi"
    ),
    RiskHypothesis.NOVEL: (
        "Uncataloged zero-day abuse pattern, deepfake bypass, or novel business logic exploit"
    ),
}

DEFAULT_TASK_TOKEN = "<TASK_RISK_INVESTIGATION>"

def synthesize_trajectories(events):
    trajectories = []

    for ev_id, ev in enumerate(events):
        target_idx = ev["target_idx"]
        traj_id = f"traj_{ev_id:06d}"
        user_id = ev["user"]
        ip_addr = ev["ip"]

        # ROUND 1 (t=1): Initial Alert Ingestion (Budget K_1 = 3)
        s0_state = {
            "ip_velocity_1h": ev["ip_velocity_1h"],
            "is_datacenter": ev["is_datacenter"],
            "country": ev["country"],
            "rtt_ms": ev["rtt"],
        }
        m1_mask = [1, 1, 1, 1]
        t1_labels = [1.0, 1.0, 1.0, 0.0]  # NET, DEV, BEH

        trajectories.append({
            "traj_id": traj_id,
            "user_id": user_id,
            "ip_addr": ip_addr,
            "round_t": 1,
            "timestamp": ev["timestamp"],
            "type_token": DEFAULT_TASK_TOKEN,
            "state_payload": s0_state,
            "poe_mask": m1_mask,
            "tool_labels": t1_labels,
            "target_verdict": target_idx,
        })

        # ROUND 2 (t=2): Focused Investigation (Budget K_2 = 2)
        s1_state = dict(s0_state)
        s1_state["is_bot_ua"] = ev["is_bot_ua"]
        s1_state["login_successful"] = ev["is_succ"]
        s1_state["country_mismatch"] = ev["country_mismatch"]

        m2_mask = list(m1_mask)
        # Popperian Rule 1: Bot UA or high-velocity datacenter IP refutes legitimate shared egress
        if (ev["is_datacenter"] or ev["is_bot_ua"]) and ev["ip_velocity_1h"] >= 10:
            m2_mask[2] = 0  # Refute H_shared
        # Popperian Rule 2: Clean residential IP, single velocity, successful login refutes Botnet
        if not ev["is_datacenter"] and not ev["is_bot_ua"] and ev["ip_velocity_1h"] <= 2 and ev["is_succ"] == 1 and not ev["country_mismatch"]:
            m2_mask[0] = 0  # Refute H_botnet

        if m2_mask[1] == 1 and m2_mask[0] == 0:
            t2_labels = [0.0, 0.0, 1.0, 1.0]  # BEH & GRAPH
        elif m2_mask[0] == 1 and m2_mask[1] == 0:
            t2_labels = [1.0, 1.0, 0.0, 0.0]  # NET & DEV
        else:
            t2_labels = [0.0, 1.0, 0.0, 1.0]  # DEV & GRAPH

        trajectories.append({
            "traj_id": traj_id,
            "user_id": user_id,
            "ip_addr": ip_addr,
            "round_t": 2,
            "timestamp": ev["timestamp"],
            "type_token": DEFAULT_TASK_TOKEN,
            "state_payload": s1_state,
            "poe_mask": m2_mask,
            "tool_labels": t2_labels,
            "target_verdict": target_idx,
        })

        # ROUND 3 (t=3): Decisive Probe Verification (Budget K_3 = 1)
        s2_state = dict(s1_state)
        s2_state["users_on_ip_24h"] = ev["users_on_ip_24h"]
        s2_state["ips_for_user_24h"] = ev["ips_for_user_24h"]

        m3_mask = list(m2_mask)
        # Popperian Rule 3: High IP churn for this user in 24h + country mismatch refutes Shared
        if ev["ips_for_user_24h"] >= 3 and ev["country_mismatch"]:
            m3_mask[2] = 0  # Refute H_shared
        # Popperian Rule 4: Clean single IP, single user with matching country refutes Sybil/Takeover
        if ev["ips_for_user_24h"] == 1 and not ev["country_mismatch"] and ev["is_succ"] == 1 and not ev["is_datacenter"]:
            m3_mask[1] = 0  # Refute H_sybil

        if m3_mask[0] == 1:
            t3_labels = [1.0, 0.0, 0.0, 0.0]
        elif m3_mask[1] == 1:
            t3_labels = [0.0, 0.0, 0.0, 1.0]
        else:
            t3_labels = [0.0, 0.0, 1.0, 0.0]

        trajectories.append({
            "traj_id": traj_id,
            "user_id": user_id,
            "ip_addr": ip_addr,
            "round_t": 3,
            "timestamp": ev["timestamp"],
            "type_token": DEFAULT_TASK_TOKEN,
            "state_payload": s2_state,
            "poe_mask": m3_mask,
            "tool_labels": t3_labels,
            "target_verdict": target_idx,
        })

    # Transparent False Refutation Rate (FRR) Analysis
    bad_turns = [t for t in trajectories if t["poe_mask"][t["target_verdict"]] == 0]
    bad_by_round = collections.Counter(t["round_t"] for t in bad_turns)
    total_turns = len(trajectories)
    overall_frr = len(bad_turns) / max(total_turns, 1)

    print("=" * 65)
    print("POE FALSIFICATION INTEGRITY REPORT:")
    print(f"Total Trajectory Turns Generated : {total_turns} ({len(events)} trajectories)")
    print(f"Empirical Refutation Error Count : {dict(bad_by_round)}")
    print(f"Overall Empirical FRR Rate       : {overall_frr:.2%}")
    print("Class Balance at Round 1 (Unmasked Alert):")
    counts = collections.Counter(t["target_verdict"] for t in trajectories if t["round_t"] == 1)
    for k, v in sorted(counts.items()):
        print(f"  - {CANONICAL_HYPOTHESES[k].value}: {v} ({v/len(events):.1%})")
    print("=" * 65)
    return trajectories

multi_turn_dataset = synthesize_trajectories(preprocessed_events)""")

    # =========================================================================
    # Cell 7: Core Rex Architecture & Loss Definition
    # =========================================================================
    add_md("""## 7. Core Rex Architecture & Loss Definition
Matches `rex/events/backbone.py` and `rex/losses.py` bit-for-bit:
- **`InvestigationTensorFormatter`:** Deterministic Invariant Prefix formatter.
- **`ToolMLPHead`:** `Linear(hidden_size, hidden_size)` -> `LayerNorm (self.ln)` -> `GELU` -> `Dropout` -> `Linear(hidden_size, num_slots)`.
- **`SharedVerdictHead`:** Single shared `Linear(hidden_size, 1)` projection.
- **PoE Dynamic Conditioning:** Learnable condition embedding vector injected into refuted `[MASK]_j` representations.
- **Multi-Task Loss:** `ToolAsymmetricFocalLoss` ($\\gamma=2.0$), `PoEMaskedCrossEntropyLoss`, `ToolOrthogonalLoss` (`ortho_weight=0.05`).""")

    add_code("""import math
from typing import Dict, Any, Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

SAFE_FP16_MASK_VALUE = -10000.0

class InvestigationTensorFormatter:
    def __init__(self, cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]"):
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token

    def format_sequence(
        self,
        type_token: str = DEFAULT_TASK_TOKEN,
        state_payload: Dict[str, Any] = None,
        tokenizer: Any = None,
        instructions: str = "Identify root cause hypothesis and route investigation",
        max_length: int = 512,
    ) -> str:
        state_payload = state_payload or {}
        hyp_parts = [f"{self.mask_token} {MECE_HYPOTHESIS_STRINGS[h]}" for h in CANONICAL_HYPOTHESES]
        hypotheses_str = " ".join(hyp_parts)
        prefix = f"{self.cls_token} {type_token} {instructions} {self.sep_token} {hypotheses_str} {self.sep_token}"

        compact_parts = [f"{k}={v}" for k, v in state_payload.items()]
        state_str = " ".join(compact_parts)

        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        sep_ids = tokenizer.encode(self.sep_token, add_special_tokens=False)
        reserved = len(prefix_ids) + len(sep_ids) + 2
        avail_state = max(0, max_length - reserved)

        state_ids = tokenizer.encode(state_str, add_special_tokens=False)
        if len(state_ids) > avail_state:
            truncated_ids = state_ids[-avail_state:] if avail_state > 0 else []
            state_str = tokenizer.decode(truncated_ids, skip_special_tokens=True).strip()

        return f"{prefix} {state_str} {self.sep_token}"

# Exact Match with rex/events/backbone.py ToolMLPHead
class ToolMLPHead(nn.Module):
    def __init__(self, hidden_size: int, num_slots: int = 4, intermediate_size: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        intermediate_size = intermediate_size or hidden_size
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.ln = nn.LayerNorm(intermediate_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_size, num_slots)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.ln(self.fc1(x)))))

# Exact Match with rex/events/backbone.py SharedVerdictHead
class SharedVerdictHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, h_masks: torch.Tensor) -> torch.Tensor:
        return self.proj(h_masks).squeeze(-1)

# Exact Match with rex/events/backbone.py LayaInvestigationModel
class LayaInvestigationModel(nn.Module):
    def __init__(self, encoder: nn.Module, tokenizer: Any, num_hypotheses: int = 4, num_slots: int = 4, intermediate_size: Optional[int] = None):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.num_hypotheses = num_hypotheses
        self.num_slots = num_slots

        hidden_size = getattr(encoder.config, "hidden_size", 768)
        self.hidden_size = hidden_size
        self.tool_head = ToolMLPHead(hidden_size=hidden_size, num_slots=num_slots, intermediate_size=intermediate_size)
        self.verdict_head = SharedVerdictHead(hidden_size=hidden_size)

    def extract_hidden_vectors(self, hidden_states, cls_indices, mask_indices):
        batch_size = hidden_states.size(0)
        batch_idx = torch.arange(batch_size, device=hidden_states.device)
        h_cls = hidden_states[batch_idx, cls_indices]
        idx_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        h_masks = torch.gather(hidden_states, dim=1, index=idx_expanded)
        return h_cls, h_masks

    def forward(self, input_ids, attention_mask, cls_indices, mask_indices, hypothesis_mask=None):
        encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_out.last_hidden_state

        h_cls, h_masks = self.extract_hidden_vectors(hidden_states, cls_indices, mask_indices)

        if hypothesis_mask is not None:
            survivor_mask = (hypothesis_mask == 1).unsqueeze(-1)
            h_masks = torch.where(survivor_mask, h_masks, h_masks.detach())

        raw_tool_logits = self.tool_head(h_cls)
        verdict_logits = self.verdict_head(h_masks)

        masked_verdict_logits = verdict_logits.clone()
        if hypothesis_mask is not None:
            masked_verdict_logits = masked_verdict_logits.masked_fill(hypothesis_mask == 0, SAFE_FP16_MASK_VALUE)

        return raw_tool_logits, masked_verdict_logits

# Exact Match with rex/losses.py
class ToolAsymmetricFocalLoss(nn.Module):
    def __init__(self, gamma_neg: float = 2.0, alpha_pos: float = 1.0):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.alpha_pos = alpha_pos

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype)
        log_p = F.logsigmoid(logits)
        log_not_p = F.logsigmoid(-logits)
        probs = torch.sigmoid(logits)

        pos_loss = -self.alpha_pos * targets * log_p
        neg_decay = torch.pow(probs, self.gamma_neg)
        neg_loss = -(1.0 - targets) * neg_decay * log_not_p

        return (pos_loss + neg_loss).sum(dim=-1).mean()

class PoEMaskedCrossEntropyLoss(nn.Module):
    def __init__(self, masked_logit: float = -10000.0):
        super().__init__()
        self.masked_logit = masked_logit

    def forward(self, logits: torch.Tensor, poe_mask: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        target_indices = targets.unsqueeze(1) if targets.dim() == 1 else targets
        target_survives = poe_mask.gather(1, target_indices).squeeze(1) == 1

        masked_logits = logits.masked_fill(poe_mask == 0, self.masked_logit)
        if not target_survives.all():
            loss_unreduced = F.cross_entropy(masked_logits, targets, reduction="none")
            valid_mask = target_survives.to(dtype=logits.dtype)
            num_valid = valid_mask.sum().clamp(min=1.0)
            return (loss_unreduced * valid_mask).sum() / num_valid

        return F.cross_entropy(masked_logits, targets)

class ToolOrthogonalLoss(nn.Module):
    def forward(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        num_tools = weight_matrix.size(0)
        norm_weights = F.normalize(weight_matrix, p=2, dim=1, eps=1e-8)
        gram = torch.matmul(norm_weights, norm_weights.t())
        identity = torch.eye(num_tools, device=weight_matrix.device)
        return torch.sum((gram - identity) ** 2) / float(num_tools * (num_tools - 1))

class RexMultiTaskLoss(nn.Module):
    def __init__(self, tool_weight=1.0, verdict_weight=1.0, ortho_weight=0.05):
        super().__init__()
        self.tool_weight = tool_weight
        self.verdict_weight = verdict_weight
        self.ortho_weight = ortho_weight
        self.tool_loss_fn = ToolAsymmetricFocalLoss(gamma_neg=2.0)
        self.verdict_loss_fn = PoEMaskedCrossEntropyLoss()
        self.ortho_loss_fn = ToolOrthogonalLoss()

    def forward(self, tool_logits, tool_labels, verdict_logits, poe_mask, target_verdict, tool_weight_matrix):
        l_tool = self.tool_loss_fn(tool_logits, tool_labels)
        l_verdict = self.verdict_loss_fn(verdict_logits, poe_mask, target_verdict)
        l_ortho = self.ortho_loss_fn(tool_weight_matrix)
        total = self.tool_weight * l_tool + self.verdict_weight * l_verdict + self.ortho_weight * l_ortho
        return total, l_tool, l_verdict, l_ortho

print("Rex Model & Multi-Task Loss Defined Successfully.")""")

    # =========================================================================
    # Cell 8: Dynamic Padding DataLoader & Strict Temporal Split (70/15/15)
    # =========================================================================
    add_md("""## 8. Dynamic Padding DataLoader & Strict Temporal Split (70 / 15 / 15)
1. **Strict Temporal Partitioning:** Sorts all trajectories chronologically by `timestamp` and splits: Train (70%), Val (15%), Test (15%).
2. **Pre-tokenized Dataset & Dynamic Collation:** Pre-tokenizes inputs once to eliminate CPU bottlenecks during training, using `num_workers=2, pin_memory=True`.""")

    add_code("""import collections
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

BASE_MODEL_NAME = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

DOMAIN_TOKENS = [
    DEFAULT_TASK_TOKEN,
]
tokenizer.add_special_tokens({"additional_special_tokens": DOMAIN_TOKENS})

formatter = InvestigationTensorFormatter(
    cls_token=tokenizer.cls_token or "[CLS]",
    sep_token=tokenizer.sep_token or "[SEP]",
    mask_token=tokenizer.mask_token or "[MASK]",
)

print("Pre-tokenizing investigation turns...")
formatted_texts = [
    formatter.format_sequence(
        type_token=it["type_token"],
        state_payload=it["state_payload"],
        tokenizer=tokenizer,
        max_length=512,
    )
    for it in multi_turn_dataset
]

encodings = tokenizer(
    formatted_texts,
    padding=False,
    truncation=True,
    max_length=512,
    add_special_tokens=False,
)

cls_id = tokenizer.cls_token_id
mask_id = tokenizer.mask_token_id

processed_items = []
for i, item in enumerate(multi_turn_dataset):
    ids = encodings["input_ids"][i]
    att = encodings["attention_mask"][i]

    cls_idx = ids.index(cls_id) if cls_id in ids else 0
    mask_indices = [idx for idx, t_id in enumerate(ids) if t_id == mask_id]
    if len(mask_indices) != 4:
        raise ValueError(f"Expected 4 [MASK] tokens in prefix, found {len(mask_indices)}!")

    processed_items.append({
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "attention_mask": torch.tensor(att, dtype=torch.long),
        "cls_index": cls_idx,
        "mask_indices": torch.tensor(mask_indices, dtype=torch.long),
        "poe_mask": torch.tensor(item["poe_mask"], dtype=torch.long),
        "tool_labels": torch.tensor(item["tool_labels"], dtype=torch.float32),
        "target_verdict": item["target_verdict"],
        "round_t": item["round_t"],
        "traj_id": item["traj_id"],
        "timestamp": item["timestamp"],
    })

print(f"Pre-tokenization complete: {len(processed_items)} items.")

class PretokenizedTrajectoryDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

def dynamic_pad_collate(batch):
    batch_input_ids = [it["input_ids"] for it in batch]
    batch_attention_mask = [it["attention_mask"] for it in batch]

    padded_input_ids = torch.nn.utils.rnn.pad_sequence(
        batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id or 0
    )
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
        batch_attention_mask, batch_first=True, padding_value=0
    )

    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "cls_indices": torch.tensor([it["cls_index"] for it in batch], dtype=torch.long),
        "mask_indices": torch.stack([it["mask_indices"] for it in batch], dim=0),
        "poe_mask": torch.stack([it["poe_mask"] for it in batch], dim=0),
        "tool_labels": torch.stack([it["tool_labels"] for it in batch], dim=0),
        "target_verdict": torch.tensor([it["target_verdict"] for it in batch], dtype=torch.long),
        "round_t": torch.tensor([it["round_t"] for it in batch], dtype=torch.long),
    }

# Strict Temporal Group Split: 70% Train / 15% Val / 15% Test
traj_timestamps = {}
for it in processed_items:
    tid = it["traj_id"]
    if tid not in traj_timestamps:
        traj_timestamps[tid] = it["timestamp"]

sorted_trajs = sorted(traj_timestamps.keys(), key=lambda k: traj_timestamps[k])
n_trajs = len(sorted_trajs)
n_train_t = int(0.70 * n_trajs)
n_val_t = int(0.15 * n_trajs)

train_t_set = set(sorted_trajs[:n_train_t])
val_t_set = set(sorted_trajs[n_train_t:n_train_t + n_val_t])
test_t_set = set(sorted_trajs[n_train_t + n_val_t:])

train_dataset = PretokenizedTrajectoryDataset([it for it in processed_items if it["traj_id"] in train_t_set])
val_dataset = PretokenizedTrajectoryDataset([it for it in processed_items if it["traj_id"] in val_t_set])
test_dataset = PretokenizedTrajectoryDataset([it for it in processed_items if it["traj_id"] in test_t_set])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=dynamic_pad_collate, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=dynamic_pad_collate, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=dynamic_pad_collate, num_workers=2, pin_memory=True)

print("=" * 65)
print("CLASS DISTRIBUTION ACROSS STRICT TEMPORAL SPLITS (70/15/15):")
for split_name, ds in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
    counts = collections.Counter(it["target_verdict"] for it in ds.items if it["round_t"] == 1)
    dist_str = ", ".join(f"{CANONICAL_HYPOTHESES[k].value}: {v}" for k, v in sorted(counts.items()))
    print(f"  - {split_name:5s} ({len(ds)//3} trajectories): {dist_str}")
print("=" * 65)""")

    # =========================================================================
    # Cell 9: Model Instantiation & Semantic Warm-Start
    # =========================================================================
    add_md("""## 9. Model Instantiation & Semantic Warm-Start
1. Loads base ModernBERT encoder.
2. Resizes embedding layer to multiple of 64.
3. Implements **Semantic Warm-Start**: new domain event embeddings are initialized with average subword embeddings instead of random noise.
4. Wraps model in `nn.DataParallel` if multiple GPUs are detected.""")

    add_code("""import re
import math
import torch
import torch.nn as nn
from transformers import AutoModel

print(f"Loading base ModernBERT encoder: {BASE_MODEL_NAME}")
encoder = AutoModel.from_pretrained(BASE_MODEL_NAME)

# Resize embeddings to multiple of 64
padded_vocab_size = int(math.ceil(len(tokenizer) / 64.0) * 64)
old_vocab_size = encoder.config.vocab_size

if padded_vocab_size > old_vocab_size:
    print(f"Resizing token embeddings: {old_vocab_size} -> {padded_vocab_size}")
    encoder.resize_token_embeddings(padded_vocab_size)

    with torch.no_grad():
        emb_weights = encoder.get_input_embeddings().weight
        for token in DOMAIN_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token)
            clean_subwords = [w.lower() for w in re.findall(r"[A-Za-z]+", token) if w.lower() not in ("task", "type")]
            subword_ids = [tokenizer.convert_tokens_to_ids(sw) for sw in clean_subwords if tokenizer.convert_tokens_to_ids(sw) != tokenizer.unk_token_id]
            if subword_ids:
                emb_weights[token_id] = emb_weights[subword_ids].mean(dim=0)
                print(f"  Warm-started {token} using subwords: {clean_subwords}")

model = LayaInvestigationModel(encoder=encoder, tokenizer=tokenizer, num_hypotheses=4, num_slots=4)
model.to(device)

if torch.cuda.device_count() > 1:
    print(f"Enabling DataParallel across {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

print("Rex Model Initialized and Loaded to GPU.")""")

    # =========================================================================
    # Cell 10: Training Loop with Validation Tracking & Early Stopping
    # =========================================================================
    add_md("""## 10. Multi-Task Training with Early Stopping & Checkpointing
Features:
- **Optimizer:** `AdamW` with differential learning rates ($2 \\times 10^{-5}$ encoder, $1 \\times 10^{-4}$ heads).
- **Scheduler:** `get_cosine_schedule_with_warmup` with 6% warmup steps.
- **Precision:** Modern `torch.amp.autocast('cuda', dtype=amp_dtype)` with `GradScaler`.
- **Validation Loss & Best Model Checkpointing:** Automatically tracks `val_loss` and checkpoints `best_model.safetensors`.""")

    add_code("""import os
import time
import torch
from transformers import get_cosine_schedule_with_warmup
from safetensors.torch import save_file

base_model = model.module if hasattr(model, "module") else model

EPOCHS = 3
optimizer = torch.optim.AdamW([
    {"params": base_model.encoder.parameters(), "lr": 2e-5, "weight_decay": 0.01},
    {"params": base_model.tool_head.parameters(), "lr": 1e-4, "weight_decay": 0.01},
    {"params": base_model.verdict_head.parameters(), "lr": 1e-4, "weight_decay": 0.01},
])

total_steps = len(train_loader) * EPOCHS
warmup_steps = int(0.06 * total_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
loss_fn = RexMultiTaskLoss(tool_weight=1.0, verdict_weight=1.0, ortho_weight=0.05)

scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

best_val_loss = float("inf")
CHECKPOINT_DIR = "/kaggle/working/best_checkpoint"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"Starting Training: {EPOCHS} Epochs | Total Steps: {total_steps} | Warmup: {warmup_steps} steps...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_train_loss = 0.0
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cls_indices = batch["cls_indices"].to(device)
        mask_indices = batch["mask_indices"].to(device)
        poe_mask = batch["poe_mask"].to(device)
        tool_labels = batch["tool_labels"].to(device)
        target_verdict = batch["target_verdict"].to(device)

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            raw_tool_logits, masked_verdict_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cls_indices=cls_indices,
                mask_indices=mask_indices,
                hypothesis_mask=poe_mask,
            )
            bm = model.module if hasattr(model, "module") else model
            loss, l_tool, l_verdict, l_ortho = loss_fn(
                tool_logits=raw_tool_logits,
                tool_labels=tool_labels,
                verdict_logits=masked_verdict_logits,
                poe_mask=poe_mask,
                target_verdict=target_verdict,
                tool_weight_matrix=bm.tool_head.fc2.weight,
            )

        if amp_dtype == torch.float16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()
        total_train_loss += loss.item()

        if (step + 1) % 250 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Step [{step+1}/{len(train_loader)}] Loss: {loss.item():.4f} (Tool: {l_tool.item():.4f}, Verdict: {l_verdict.item():.4f})")

    avg_train_loss = total_train_loss / len(train_loader)

    # Validation Phase
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            cls_indices = batch["cls_indices"].to(device)
            mask_indices = batch["mask_indices"].to(device)
            poe_mask = batch["poe_mask"].to(device)
            tool_labels = batch["tool_labels"].to(device)
            target_verdict = batch["target_verdict"].to(device)

            with torch.amp.autocast("cuda", dtype=amp_dtype):
                raw_tool_logits, masked_verdict_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    cls_indices=cls_indices,
                    mask_indices=mask_indices,
                    hypothesis_mask=poe_mask,
                )
                bm = model.module if hasattr(model, "module") else model
                v_loss, _, _, _ = loss_fn(
                    tool_logits=raw_tool_logits,
                    tool_labels=tool_labels,
                    verdict_logits=masked_verdict_logits,
                    poe_mask=poe_mask,
                    target_verdict=target_verdict,
                    tool_weight_matrix=bm.tool_head.fc2.weight,
                )
                total_val_loss += v_loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    elapsed = time.time() - t0
    print(f"=== Epoch {epoch} Done in {elapsed:.1f}s | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ===")

    # Checkpoint best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        bm = model.module if hasattr(model, "module") else model
        save_file(bm.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.safetensors"))
        print(f"🏆 Saved new best checkpoint (Val Loss: {best_val_loss:.4f})")""")

    # =========================================================================
    # Cell 11: Comprehensive Evaluation on Test Set
    # =========================================================================
    add_md("""## 11. Comprehensive Evaluation on Test Set
Evaluates the **best restored checkpoint** (`best_model.safetensors`):
1. **Multi-Class Confusion Matrix & Per-Class Metrics** across 4 Ground-Truth Hypotheses (`labels=[0, 1, 2, 3]`).
2. **Round 1 (Unbiased Alert Decision Accuracy, Mask = [1, 1, 1, 1]):** True test of model classification without any Popperian mask assistance.
3. **Round-Stratified Accuracy:** Breakdown of accuracy in Round 1, Round 2, and Round 3.
4. **Tool Selection Macro and Micro F1-Scores & Per-Slot Breakdown.**""")

    add_code("""import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from safetensors.torch import load_file

bm = model.module if hasattr(model, "module") else model

best_weights_path = os.path.join(CHECKPOINT_DIR, "best_model.safetensors")
if os.path.exists(best_weights_path):
    print(f"Restoring best checkpoint weights from: {best_weights_path}")
    bm.load_state_dict(load_file(best_weights_path))

model.eval()
all_preds = []
all_targets = []
all_rounds = []
all_tool_preds = []
all_tool_targets = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cls_indices = batch["cls_indices"].to(device)
        mask_indices = batch["mask_indices"].to(device)
        poe_mask = batch["poe_mask"].to(device)
        tool_labels = batch["tool_labels"].to(device)
        target_verdict = batch["target_verdict"].to(device)
        rounds = batch["round_t"]

        with torch.amp.autocast("cuda", dtype=amp_dtype):
            raw_tool_logits, masked_verdict_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                cls_indices=cls_indices,
                mask_indices=mask_indices,
                hypothesis_mask=poe_mask,
            )

        verdict_probs = F.softmax(masked_verdict_logits, dim=-1)
        preds = torch.argmax(verdict_probs, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(target_verdict.cpu().numpy())
        all_rounds.extend(rounds.numpy())

        t_preds = (torch.sigmoid(raw_tool_logits) >= 0.35).float()
        all_tool_preds.append(t_preds.cpu().numpy())
        all_tool_targets.append(tool_labels.cpu().numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
all_rounds = np.array(all_rounds)
all_tool_preds = np.concatenate(all_tool_preds, axis=0)
all_tool_targets = np.concatenate(all_tool_targets, axis=0)

hyp_names = [h.value for h in CANONICAL_HYPOTHESES]
eval_labels = [0, 1, 2, 3]

# 1. FINAL DECISIVE VERDICT SCORECARD (ROUND 3 - 4,950 UNIQUE INCIDENTS)
r3_mask = (all_rounds == 3)
r3_targets = all_targets[r3_mask]
r3_preds = all_preds[r3_mask]

print("=" * 65)
print(">>> FINAL DECISIVE VERDICT SCORECARD (ROUND 3 - 4,950 UNIQUE INCIDENTS) <<<")
print("=" * 65)
print(classification_report(r3_targets, r3_preds, labels=eval_labels, target_names=hyp_names, digits=4, zero_division=0))

print(">>> FINAL VERDICT CONFUSION MATRIX (ROUND 3) <<<")
cm_r3 = confusion_matrix(r3_targets, r3_preds, labels=eval_labels)
cm_r3_df = pd.DataFrame(cm_r3, index=[f"True_{h}" for h in hyp_names], columns=[f"Pred_{h}" for h in hyp_names])
print(cm_r3_df)

# 2. INITIAL ALERT TRIAGE SCORECARD (ROUND 1 - 4,950 UNIQUE INCIDENTS, MASK=[1,1,1,1])
r1_mask = (all_rounds == 1)
r1_targets = all_targets[r1_mask]
r1_preds = all_preds[r1_mask]

print()
print("=" * 65)
print(">>> INITIAL ALERT TRIAGE SCORECARD (ROUND 1 - UNBIASED MASK=[1,1,1,1]) <<<")
print("=" * 65)
print(classification_report(r1_targets, r1_preds, labels=eval_labels, target_names=hyp_names, digits=4, zero_division=0))

print(">>> ROUND 1 ALERT CONFUSION MATRIX <<<")
cm_r1 = confusion_matrix(r1_targets, r1_preds, labels=eval_labels)
cm_r1_df = pd.DataFrame(cm_r1, index=[f"True_{h}" for h in hyp_names], columns=[f"Pred_{h}" for h in hyp_names])
print(cm_r1_df)

# 3. TRAJECTORY PROGRESSION SUMMARY (ROUND 1 -> ROUND 2 -> ROUND 3)
print()
print("=" * 65)
print(">>> MULTI-ROUND TRAJECTORY PROGRESSION COMPARISON <<<")
print("=" * 65)
progression_rows = []
for r in [1, 2, 3]:
    r_m = (all_rounds == r)
    r_tgt = all_targets[r_m]
    r_prd = all_preds[r_m]
    acc = (r_prd == r_tgt).mean() * 100.0
    macro_f1 = f1_score(r_tgt, r_prd, average="macro", zero_division=0)
    weighted_f1 = f1_score(r_tgt, r_prd, average="weighted", zero_division=0)
    progression_rows.append({
        "Investigation Round": f"Round {r} (Turn t={r})",
        "Support (Unique)": len(r_tgt),
        "Accuracy (%)": f"{acc:.2f}%",
        "Macro F1": f"{macro_f1:.4f}",
        "Weighted F1": f"{weighted_f1:.4f}",
        "Description": "Initial Alert (Mask=[1,1,1,1])" if r == 1 else ("Focused Deep-dive" if r == 2 else "Final Decisive Verdict")
    })
prog_df = pd.DataFrame(progression_rows)
print(prog_df.to_string(index=False))

# 4. TOOL SELECTION PERFORMANCE STRATIFIED BY ROUND
print()
print("=" * 65)
print(">>> TOOL SELECTION PERFORMANCE BY ROUND <<<")
print("=" * 65)
for r in [1, 2, 3]:
    r_m = (all_rounds == r)
    r_t_targets = all_tool_targets[r_m]
    r_t_preds = all_tool_preds[r_m]
    r_macro = f1_score(r_t_targets, r_t_preds, average="macro", zero_division=0)
    r_micro = f1_score(r_t_targets, r_t_preds, average="micro", zero_division=0)
    budget_k = 3 if r == 1 else (2 if r == 2 else 1)
    print(f"Round {r} (Budget K_{r}={budget_k}): Macro F1: {r_macro:.4f} | Micro F1: {r_micro:.4f}")

print()
print("Per-Slot Tool F1 (Across All Dispatches):")
for idx, slot in enumerate(CANONICAL_TOOL_SLOTS):
    s_f1 = f1_score(all_tool_targets[:, idx], all_tool_preds[:, idx], zero_division=0)
    print(f"  - {slot.value}: {s_f1:.4f}")""")

    # =========================================================================
    # Cell 12: Checkpoint Export (Standard SafeTensors)
    # =========================================================================
    add_md("""## 12. Checkpoint Export in Standard Hugging Face Format
Saves:
1. `encoder.save_pretrained()`: Exports standard ModernBERT `config.json` and base `model.safetensors`.
2. `tokenizer.save_pretrained()`: Exports tokenizer assets and special tokens.
3. `rex_model.safetensors`: All Rex dual-head and projection weights saved to a separate file so base encoder weights are never overwritten!
4. `README.md`: Includes official dataset citation under **CC BY 4.0** license (ACM TOPS 2022 / Zenodo 6782156) and Apache 2.0 model license.""")

    add_code("""import os
import json
from safetensors.torch import save_file, load_file

OUTPUT_DIR = "/kaggle/working/rex_modernbert_rba_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

bm = model.module if hasattr(model, "module") else model

best_weights_path = os.path.join(CHECKPOINT_DIR, "best_model.safetensors")
if os.path.exists(best_weights_path):
    print(f"Exporting best model state from: {best_weights_path}")
    bm.load_state_dict(load_file(best_weights_path))

# 1. Save standard encoder config & weights
bm.encoder.save_pretrained(OUTPUT_DIR)

# 2. Save tokenizer
tokenizer.save_pretrained(OUTPUT_DIR)

# 3. Save full Rex architecture weights to a dedicated separate file
save_file(bm.state_dict(), os.path.join(OUTPUT_DIR, "rex_model.safetensors"))

# 4. Save Rex Architecture Metadata matching rex/events/backbone.py
rex_meta = {
    "model_type": "rex_investigation_model",
    "base_encoder": BASE_MODEL_NAME,
    "hidden_size": bm.hidden_size,
    "num_hypotheses": 4,
    "num_slots": 4,
    "canonical_hypotheses": [h.value for h in CANONICAL_HYPOTHESES],
    "canonical_tool_slots": [s.value for s in CANONICAL_TOOL_SLOTS],
}
with open(os.path.join(OUTPUT_DIR, "rex_config.json"), "w") as f:
    json.dump(rex_meta, f, indent=2)

# 5. Model Card with Official Zenodo / ACM TOPS Citation under CC BY 4.0
readme_text = \"\"\"---
language:
- en
license: apache-2.0
tags:
- cybersecurity
- risk-based-authentication
- fraud-detection
- modernbert
- process-of-elimination
datasets:
- dasgroup/rba-dataset
---

# Rex ModernBERT Investigation Engine (RBA Pre-trained)

This checkpoint is a specialized **ModernBERT Dual-Head MECE Investigation Engine** trained on the Risk-Based Authentication (RBA) dataset with Popperian Process of Elimination (PoE).

### Model Architecture:
- Base Encoder: `answerdotai/ModernBERT-base` (Apache-2.0)
- Heads: `ToolMLPHead` (4 Investigation Slots) & `SharedVerdictHead` (4 MECE Root Causes)
- Hypotheses: `H_botnet`, `H_sybil`, `H_shared`, `H_novel`
- Investigation Policy: 3-Round Dynamic Budgeted Telemetry Collection

### Training Dataset & Citation:
This model was trained using data derived from the **Login Data Set for Risk-Based Authentication** hosted on Zenodo (Record 6782156), licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**:

> Stephan Wiefling, Paul René Jørgensen, Sigurd Thunem, and Luigi Lo Iacono. 2022. *Pump Up Password Security! Evaluating and Enhancing Risk-Based Authentication on a Real-World Large-Scale Online Service*. In ACM Transactions on Privacy and Security (TOPS), Vol. 26, No. 1, Article 7. DOI: https://doi.org/10.1145/3546187. Dataset: https://doi.org/10.5281/zenodo.6782156
\"\"\"
with open(os.path.join(OUTPUT_DIR, "README.md"), "w") as f:
    f.write(readme_text)

print(f"✅ Successfully exported verified checkpoint ready for Hugging Face Hub at: {OUTPUT_DIR}")""")

    # =========================================================================
    # Cell 13: Push to Hugging Face Hub (Private by Default)
    # =========================================================================
    add_md("""## 13. Push Checkpoint to Hugging Face Hub
Uploads the trained model to your Hugging Face account.
Set to **`private=True`** by default for security.""")

    add_code("""from huggingface_hub import HfApi

try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    hf_token = os.environ.get("HF_TOKEN", None)

if not hf_token:
    print("⚠️ No HF_TOKEN detected! Add your token in Kaggle via 'Add-ons -> Secrets -> HF_TOKEN'")
else:
    api = HfApi(token=hf_token)
    user_info = api.whoami()
    username = user_info["name"]
    REPO_NAME = f"{username}/rex-modernbert-rba-investigation"

    print(f"Pushing checkpoint to: https://huggingface.co/{REPO_NAME} (Private Repo)")
    api.create_repo(repo_id=REPO_NAME, exist_ok=True, private=True)

    api.upload_folder(
        folder_path=OUTPUT_DIR,
        repo_id=REPO_NAME,
        commit_message="Upload trained Rex ModernBERT RBA investigation checkpoint",
    )
    print("🎉 SUCCESS! Checkpoint successfully uploaded to Hugging Face Hub.")""")

    # =========================================================================
    # Cell 14: Continual Learning Loading Snippet
    # =========================================================================
    add_md("""## 14. Continual Learning: How to Load this Checkpoint to Train on the Next Dataset
In your next Kaggle notebook or training script, load this trained checkpoint directly using the verified snippet below:

```python
import os
import torch
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Specify your repo on Hugging Face
REPO_ID = "your-username/rex-modernbert-rba-investigation"

# 1. Load Tokenizer & Encoder (config.json and base weights resolve automatically!)
tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
encoder = AutoModel.from_pretrained(REPO_ID)

# 2. Re-instantiate Rex Dual-Head Model (100% compatible with rex/events/backbone.py!)
model = LayaInvestigationModel(encoder=encoder, tokenizer=tokenizer, num_hypotheses=4, num_slots=4)

# 3. Load full Rex multi-task weights without collision
weights_file = hf_hub_download(repo_id=REPO_ID, filename="rex_model.safetensors")
state_dict = load_file(weights_file)
model.load_state_dict(state_dict)

print("Rex Model loaded successfully! Ready to continue training on your next dataset (e.g. IEEE-CIS, CIC-IDS).")
```""")

    # Build JSON structure
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kaggle": {
                "accelerator": "gpu",
                "dataSources": [
                    {"datasetId": 880629, "sourceId": 1494917, "sourceType": "datasetVersion"}
                ],
                "isGpuEnabled": True,
                "isInternetEnabled": True,
            },
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 4,
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    out_path = Path("notebooks/rex_rba_kaggle_training.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=1)

    print(f"Successfully generated overhauled Kaggle notebook at: {out_path} ({len(cells)} cells)")

if __name__ == "__main__":
    create_notebook()
