# Training Rex

Rex (`arch = "rex2"`) adds investigation heads to a Laya checkpoint: sufficiency S, risk R, a tool matcher, a tool gain head, a hypothesis scorer, a latent ponder block and a verdict bridge. `scripts/train_rex.py` trains them on multi-round trajectories; the Laya loop in `rex.train` only trains the verdict path and warns when it is given a Rex model.

## 1. Data

`scripts/generate_rex_data.py` simulates cases from a registry (`examples/rex/registry.example.json`) and writes `train.jsonl` / `val.jsonl`. Each line is one decision step; the fields and label semantics are in [the label schema](rex-label-schema.md).

The registry declares tools (`level`, `cost`, returned metrics), domains, hypotheses, falsification rules and the business risk of each hypothesis (`risk`). The risk label of a case is the risk of its true cause. The model scores the risk of every hypothesis from its marker and reports their average under the verdict distribution, so R drops once the evidence points to a benign cause.

## 2. Training

```bash
python scripts/generate_rex_data.py --registry examples/rex/registry.example.json --out-dir data/rex
python scripts/train_rex.py --base convaiinnovations/laya --registry examples/rex/registry.example.json \
    --train data/rex/train.jsonl --val data/rex/val.jsonl --stage heads --out runs/rex-heads
python scripts/train_rex.py --base runs/rex-heads --registry examples/rex/registry.example.json \
    --train data/rex/train.jsonl --val data/rex/val.jsonl --stage ponder --out runs/rex-ponder
```

A Laya base is converted with the registry's tools and type tokens; its choice temperature becomes the Rex verdict temperature. The registry's tool specs (level, cost) and the encoded tool descriptions are stored in the checkpoint.

Each item is a whole case. Round 1 runs `forward(state=None)` on the opening layout; every later round runs `forward(state, evidence_new)` on the gold observations of the previous round, as `InvestigatorAgent` and the Part B `RexAdapter` do at inference. The state carried across rounds is detached between rounds unless `--no-detach-between-rounds` is given.

| Stage | Trained |
| :--- | :--- |
| `heads` | `head_s`, `head_r`, `tool_matcher`, `tool_gain`, `hyp_scorer`, `alive_count_emb` |
| `ponder` | the above plus `ponder_block` and `verdict_bridge` |
| `full` | every parameter; the encoder uses `--lr-encoder` |

### Objective

`RexMultiTaskLoss` sums, over every executed case-round, the losses weighted by `RexLossWeights`:

- `ToolAsymmetricFocalLoss` on the tool matcher against the teacher tools;
- `PoEMaskedCrossEntropyLoss` on the verdict over the surviving hypotheses;
- `ToolOrthogonalLoss` on the tool projections;
- binary cross-entropy for S, R, the gain head (masked where the gain is unobserved or the tool redundant) and the hypothesis scorer.

`RexMultiTaskInputs` carries the tensors; `LossBreakdown` returns the total and every component. STALL has no loss: the dispatcher derives it from S, the gains and the remaining tools, and the `stall_star` label is used for evaluation.

## 3. Calibration and evaluation

After training, the validation cases are split in two by case (or `calib_samples` is given to `rex.rex_train.train_rex`):

1. the calibration half fits the verdict temperature, then the conformal threshold `q_hat` on the temperature-scaled probabilities; both are stored in the checkpoint;
2. the other half is the reported evaluation, so no reported number comes from the cases that fitted the calibration.

`rex.rex_train.evaluate` reports the validation loss, final-round verdict accuracy and:

| Metric | Meaning |
| :--- | :--- |
| `s_auroc`, `s_ece` | Sufficiency head discrimination and calibration over all steps |
| `s_auroc_within_round` | The same AUROC computed inside each round index; a head that only counts rounds scores 0.5 |
| `s_mean_round_k` | Mean S per round |
| `r_mae`, `r_auroc_high_risk` | Risk head error, and how well it separates risk above the 0.30 threshold |
| `gain_auroc` | Gain head on callable, observed tools |
| `hyp_auroc` | Hypothesis scorer against survival |
| `stall_rate_*`, `stall_precision`, `stall_recall` | Dispatcher-derived STALL against `stall_star` |
| `conformal_coverage`, `conformal_mean_set_size` | Coverage of the conformal sets at `q_hat`, and their size |

When final-round accuracy is below $1 - \alpha$ (90 % by default), APS needs full sets to reach the coverage, so `q_hat` reaches 1.0 and the conformal auto-commit never fires. That is why `StopConfig.accept_conformal_verdict` stays off until the threshold is calibrated on real cases.

## 4. Inference

`InvestigatorAgent` runs a case end to end. Before round 1 it applies the falsification rules to the alert metrics (`initial_observations`), as the training rollouts do. After each round it appends the observations to memory (`ingest_evidence`), scores the hypotheses with the hypothesis scorer on that memory, and combines the scores with the rules in `PoEFalsificationEngine` (via the hybrid engine), which keeps rule-eliminated hypotheses eliminated. It stops on a `VERDICT` or `STALL` command, when every cataloged hypothesis is eliminated, or after `max_rounds`, and returns an `InvestigationResult` with one `RoundTrace` per round and a `TerminationReason`. A STALL, a spent budget without sufficient evidence, a risk above the threshold or an `H_novel` verdict is escalated.

Falsification rules (`FalsificationRule`, operators in `ComparisonOperator`) live in `rex_poe` and the registry; `HypothesisDefinition` is the in-memory hypothesis contract. Input layout and evidence formatting are in `rex_format`, and type tokens are added to the tokenizer with `rex_vocab`.

`RexDecisionModel.rex_forward` is a legacy shim: it scores hypotheses from the `[CLS]` vector without the ponder block or the evidence memory, so it is not the inference path; use `forward` (or `InvestigatorAgent`).
