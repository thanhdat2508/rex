"""Train the Rex decision heads on multi-round investigation trajectories (T4).

Starts from a Laya checkpoint (converted with the registry's tools and type tokens) or from an
existing Rex checkpoint, trains one freeze stage, and saves a Rex checkpoint:

    python scripts/train_rex.py --base convaiinnovations/laya --registry examples/rex/registry.example.json \\
        --train data/rex/train.jsonl --val data/rex/val.jsonl --stage heads --out runs/rex-heads
    python scripts/train_rex.py --base runs/rex-heads --registry examples/rex/registry.example.json \\
        --train data/rex/train.jsonl --val data/rex/val.jsonl --stage ponder --out runs/rex-ponder

Without --train, cases are simulated from the registry (--n-cases, split by case).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rex.losses import RexLossWeights
from rex.rex_data import generate, read_jsonl, split_by_case
from rex.rex_model import RexDecisionModel, from_rex, save_rex_checkpoint
from rex.rex_registry import load_registry
from rex.rex_train import STAGES, RexTrainConfig, train_rex
from rex.rex_vocab import add_type_tokens
from rex.train import load_checkpoint


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train Rex heads on trajectory data.")
    p.add_argument("--base", required=True, help="Laya or Rex checkpoint directory / Hub ID.")
    p.add_argument("--registry", required=True, help="Rex registry JSON/YAML (tools, domains, hypotheses).")
    p.add_argument("--out", required=True, help="Output directory for the trained Rex checkpoint.")
    p.add_argument("--train", default=None, help="train.jsonl from scripts/generate_rex_data.py.")
    p.add_argument("--val", default=None, help="val.jsonl from scripts/generate_rex_data.py.")
    p.add_argument("--n-cases", type=int, default=2000, help="Cases to simulate when --train is not given.")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--stage", choices=STAGES, default="heads")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-cases", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr-heads", type=float, default=1e-4)
    p.add_argument("--lr-encoder", type=float, default=2.5e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--no-detach-between-rounds", action="store_true",
                   help="Backpropagate through the state carried across rounds (full BPTT).")
    p.add_argument("--no-encode-tools", action="store_true",
                   help="Keep the current tool_embeddings instead of encoding registry tool descriptions.")
    p.add_argument("--gain-weight", type=float, default=1.0)
    p.add_argument("--hyp-weight", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", help="CUDA autocast (bf16 when supported, else fp16).")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def load_base(base: str, registry):
    """A RexDecisionModel for `base`: loaded as-is when it is a Rex checkpoint, converted when it is Laya."""
    model, tok, cfg = load_checkpoint(base)
    if isinstance(model, RexDecisionModel):
        registry.check_model_tools(model.tools)
        return model, tok, cfg
    if registry.type_tokens:
        add_type_tokens(tok, model, registry.type_tokens)
    rex_model = from_rex(
        model,
        tools=registry.tools,
        type_tokens=registry.type_tokens,
        tool_hidden=registry.tool_hidden,
        tool_dropout=registry.tool_dropout,
        tool_prior=registry.tool_prior,
        base_cfg=cfg,
    )
    return rex_model, tok, cfg


def main(argv=None):
    args = parse_args(argv)
    registry = load_registry(args.registry)
    model, tok, base_cfg = load_base(args.base, registry)

    if args.train:
        train_samples = read_jsonl(args.train)
        val_samples = read_jsonl(args.val) if args.val else []
    else:
        samples = generate(registry, args.n_cases, args.seed)
        train_samples, val_samples = split_by_case(samples, val_ratio=args.val_ratio, seed=args.seed)

    config = RexTrainConfig(
        stage=args.stage,
        epochs=args.epochs,
        batch_cases=args.batch_cases,
        grad_accum=args.grad_accum,
        lr_heads=args.lr_heads,
        lr_encoder=args.lr_encoder,
        warmup_ratio=args.warmup_ratio,
        detach_between_rounds=not args.no_detach_between_rounds,
        amp=args.amp,
        seed=args.seed,
        loss_weights=RexLossWeights(gain_weight=args.gain_weight, hyp_weight=args.hyp_weight),
    )
    device = torch.device(args.device) if args.device else None
    summary = train_rex(model, tok, registry, train_samples, val_samples, config=config, device=device,
                        encode_tools=not args.no_encode_tools)

    save_rex_checkpoint(model.cpu(), tok, args.out, base_cfg=base_cfg)
    with open(os.path.join(args.out, "rex_train_summary.json"), "w", encoding="utf-8") as f:
        json.dump({**{k: v for k, v in summary.items() if k != "grad_norms"}, "stage": args.stage}, f, indent=2)
    print(f"Saved Rex checkpoint to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
