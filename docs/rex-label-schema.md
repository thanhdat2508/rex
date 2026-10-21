# Rex Training Label & Dataset Schema (Collaborator F)

This document formalizes the JSON schema for multi-turn training samples and counterfactual supervision labels used to train the Rex unified architecture.

---

## 1. Top-Level Sample Format (JSON Lines record)

Each row in `train.jsonl` or `val.jsonl` represents a single decision step $(case, round)$ within an incident exploration trajectory.

```json
{
  "case_id": "incident_payment_fraud_88291",
  "domain_key": "<TYPE_PAYMENT_FRAUD>",
  "round_idx": 2,
  "instruction": "Investigate unusual velocity and card fingerprint mismatch on account",
  "hypotheses": {
    "H_credential_stuffing": "Account takeover via compromised credential stuffing",
    "H_friendly_fraud": "Dispute initiated by family member or legitimate cardholder",
    "H_stolen_instrument": "Purchase using stolen credit card data",
    "H_novel": "Unexplained anomalous transaction pattern"
  },
  "history": [
    {
      "round": 1,
      "tools_called": ["tool_card_velocity", "tool_device_profile"],
      "observations": {
        "card_velocity_1h": 8,
        "is_proxy_vpn": true
      }
    }
  ],
  "new_evidence": "device_canvas_hash=39281a is_rooted=false",
  "surviving_hypotheses": ["H_stolen_instrument", "H_novel"],
  "labels": {
    "sufficient": 0,
    "target_verdict": 2,
    "risk_label": 0.25,
    "tool_gains": {
      "tool_geolocation": 1,
      "tool_merchant_history": 0,
      "tool_card_velocity": 0,
      "tool_biometrics": -100
    },
    "teacher_tools": ["tool_geolocation"],
    "is_stall": 0,
    "key_facts_remaining": ["buyer_country_mismatch"]
  }
}
```

---

## 2. Field Definitions

| Field | Type | Description |
| :--- | :--- | :--- |
| `case_id` | `str` | Unique incident identifier. |
| `round_idx` | `int` | Turn index (1, 2, 3). |
| `history` | `list` | Previous rounds' called tools and structured observations. |
| `new_evidence` | `str` | Text representation of observations gathered in the current turn ($\le 64$ tokens). |
| `labels.sufficient` | `int (0/1)` | Binary ground truth: whether all key facts necessary for conclusion are in memory. |
| `labels.target_verdict` | `int` | Index of the true underlying hypothesis among surviving slots. |
| `labels.risk_label` | `float` | Estimated operational/financial penalty score in $[0, 1]$. |
| `labels.tool_gains` | `dict[str, int]` | Expected information gain per tool: `1` if tool uncovers an unseen key fact or flips wrong verdict; `0` if tool yields no new information; `-100` if tool observation is unrecorded/unobserved (masked from loss). |
| `labels.teacher_tools` | `list[str]` | Tools selected by human experts or optimal teacher policy. |
| `labels.is_stall` | `int (0/1)` | `1` if $S=0$ and no remaining allowed tool has $\text{gain}=1$. |
