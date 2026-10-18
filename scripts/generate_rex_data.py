"""Command-line utility to generate Rex multi-turn trajectory datasets (Part 3)."""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.rex_data import (
    RexTrajectoryDataset,
    collate_rex_train,
    generate,
    split_by_case,
    stats,
    write_jsonl,
)
from rex.rex_registry import load_registry


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Rex multi-turn investigation trajectory datasets."
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="examples/rex/registry.example.json",
        help="Path to Rex registry JSON/YAML file.",
    )
    parser.add_argument(
        "--n-cases",
        type=int,
        default=20000,
        help="Number of independent incident cases to simulate.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/rex",
        help="Output directory to save train.jsonl, val.jsonl, and stats.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible rollout generation.",
    )
    parser.add_argument(
        "--novel-ratio",
        type=float,
        default=0.12,
        help="Fraction of cases sampled from OOD / H_novel distribution (10-15%%).",
    )
    parser.add_argument(
        "--p-falsify",
        type=float,
        default=0.85,
        help="Probability of generating metrics that trigger falsification rules for non-target hypotheses.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Fraction of incident cases held out for validation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to Rex checkpoint directory to validate tools and vocabulary compatibility.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print(" Rex Trajectory Dataset Generator (Part 3)")
    print("=" * 70)

    # 1. Load and validate registry
    print(f"Loading registry from: {args.registry}")
    registry = load_registry(args.registry)
    print(f"  - Tools registered: {len(registry.tools)}")
    print(f"  - Domains registered: {len(registry.domains)} ({list(registry.domains.keys())})")
    print(f"  - Metrics registered: {len(registry.metrics)}")

    # 2. Checkpoint validation if provided
    model = None
    tok = None
    cfg = None
    if args.checkpoint:
        print(f"\nValidating against Rex checkpoint: {args.checkpoint}")
        from rex.train import load_checkpoint
        model, tok, cfg = load_checkpoint(args.checkpoint)

        # Check tool head alignment
        registry.check_model_tools(model.tools)
        print("  - Tools match model.tools in exact count and vocabulary.")

        # Check domain type tokens in tokenizer
        for d_key in registry.domains:
            tid = tok.convert_tokens_to_ids(d_key)
            if tid == tok.unk_token_id:
                raise ValueError(
                    f"Domain token '{d_key}' is mapped to [UNK] in checkpoint tokenizer. "
                    f"Ensure type tokens were added via rex_vocab.add_type_tokens."
                )
        print("  - All domain type tokens are registered in checkpoint tokenizer.")

    # 3. Generate rollouts
    print(f"\nSimulating {args.n_cases} incident cases (seed={args.seed}, novel_ratio={args.novel_ratio})...")
    samples = generate(
        registry=registry,
        n_cases=args.n_cases,
        rng_or_seed=args.seed,
        novel_ratio=args.novel_ratio,
        p_falsify=args.p_falsify,
    )
    print(f"  - Generated {len(samples)} total trajectory step samples.")

    # 4. Split by case strictly
    train_samples, val_samples = split_by_case(
        samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"  - Split into {len(train_samples)} train samples and {len(val_samples)} val samples.")

    # 5. Compute statistics
    dataset_stats = stats(samples, registry)
    dataset_stats["train_samples"] = len(train_samples)
    dataset_stats["val_samples"] = len(val_samples)

    # 6. Save outputs
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"
    stats_path = out_dir / "stats.json"

    print(f"\nWriting datasets to {out_dir}:")
    write_jsonl(train_path, train_samples)
    print(f"  - Wrote {train_path}")
    write_jsonl(val_path, val_samples)
    print(f"  - Wrote {val_path}")
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(dataset_stats, f, indent=2)
    print(f"  - Wrote {stats_path}")

    # 7. Optional dry-run batch validation with checkpoint
    if model is not None and tok is not None:
        print("\nExecuting dry-run batch through model.rex_forward...")
        ds_val = RexTrajectoryDataset(
            samples=val_samples[:16],
            tok=tok,
            tools=model.tools,
            registry=registry,
            max_len=cfg.get("max_len", 512),
            head_max_len=cfg.get("head_max_len", 192),
            shuffle_hypotheses=True,
        )
        batch_items = [ds_val[i] for i in range(min(len(ds_val), 16))]
        b = collate_rex_train(batch_items, tok.pad_token_id)

        model.eval()
        with torch.no_grad():
            out = model.rex_forward(
                input_ids=b["input_ids"],
                attention_mask=b["attention_mask"],
                marker_pos=b["marker_pos"],
                marker_mask=b["marker_mask"],
                qtype=b["qtype"],
                hyp_alive=b["hyp_alive"],
                tool_mask=b.get("tool_mask"),
            )
        print("  - Dry-run forward pass successful!")
        print(f"  - Verdict logits shape: {list(out['verdict_logits'].shape)}")
        print(f"  - Tool logits shape:    {list(out['tool_logits'].shape)}")

    # 8. Summary display
    print("\n" + "=" * 70)
    print(" Dataset Statistics Summary")
    print("=" * 70)
    print(f" Total incident cases:              {dataset_stats['n_cases']}")
    print(f" Total step samples:                {dataset_stats['n_samples']}")
    print(f" Average steps per case:            {dataset_stats['n_samples'] / max(1, dataset_stats['n_cases']):.2f}")
    print(f" OOD / H_novel case ratio:          {dataset_stats['novel_ratio'] * 100:.1f}%")
    print(f" Novel cases with all known falsified: {dataset_stats['novel_known_eliminated_rate'] * 100:.1f}%")
    print(" Average tools called per round:")
    for r, avg in dataset_stats["avg_tools_per_round"].items():
        print(f"   {r}: {avg:.2f}")
    print("=" * 70)
    print("Pipeline completed successfully.\n")


if __name__ == "__main__":
    main()
