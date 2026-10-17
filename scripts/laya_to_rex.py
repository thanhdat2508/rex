"""CLI tool to convert an existing Laya checkpoint into an untrained Rex checkpoint."""
import argparse
import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rex.rex_model import from_laya, save_rex_checkpoint
from rex.rex_vocab import add_type_tokens
from rex.train import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a Laya checkpoint to a Rex checkpoint.")
    parser.add_argument(
        "--laya",
        type=str,
        default="convaiinnovations/laya",
        help="Path or HuggingFace repo ID of the base Laya checkpoint.",
    )
    parser.add_argument(
        "--spec",
        type=str,
        required=True,
        help="Path to JSON file specifying tools and type_tokens.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for the new Rex checkpoint.",
    )
    parser.add_argument(
        "--pad-to-multiple-of",
        type=int,
        default=None,
        help="Optional integer (e.g. 64) to pad embedding matrix size for tensor core efficiency.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load specification
    with open(args.spec, "r", encoding="utf-8") as f:
        spec = json.load(f)

    tools = spec.get("tools", [])
    type_tokens = spec.get("type_tokens", {})
    tool_hidden = spec.get("tool_hidden")
    tool_dropout = spec.get("tool_dropout", 0.1)
    tool_prior = spec.get("tool_prior")

    print(f"Loading specification from {args.spec}...")
    print(f"  Tools ({len(tools)}): {tools}")
    print(f"  Type tokens ({len(type_tokens)}): {list(type_tokens.keys())}")

    # 2. Load trained Laya checkpoint (handles local path, Hub ID, or alias like 'multilingual')
    print(f"Loading trained Laya checkpoint from {args.laya}...")
    base_model, tok, base_cfg = load_checkpoint(args.laya)

    # 4. Add type tokens and warm-start their embeddings
    if type_tokens:
        print("Expanding tokenizer vocabulary and warm-starting token embeddings...")
        add_type_tokens(tok, base_model, type_tokens)

    if args.pad_to_multiple_of:
        embed_module = None
        for candidate in (base_model, getattr(base_model, "encoder", None)):
            if candidate is not None and hasattr(candidate, "get_input_embeddings"):
                res = candidate.get_input_embeddings()
                if res is not None and hasattr(res, "weight"):
                    embed_module = res
                    break
        cur_size = embed_module.weight.size(0) if embed_module is not None else len(tok)
        needed_size = max(cur_size, len(tok))
        rem = needed_size % args.pad_to_multiple_of
        target_size = needed_size if rem == 0 else needed_size + (args.pad_to_multiple_of - rem)
        if target_size > cur_size:
            print(f"Padding embedding size from {cur_size} to multiple of {args.pad_to_multiple_of}: {target_size}")
            base_model.encoder.resize_token_embeddings(target_size)

    # 5. Convert to RexDecisionModel with new ToolHead
    print("Converting DecisionModel to RexDecisionModel...")
    rex_model = from_laya(
        base_model,
        tools=tools,
        type_tokens=type_tokens,
        tool_hidden=tool_hidden,
        tool_dropout=tool_dropout,
        tool_prior=tool_prior,
    )

    # 6. Save checkpoint
    print(f"Saving Rex checkpoint to {args.out}...")
    save_rex_checkpoint(rex_model, tok, args.out, base_cfg=base_cfg)

    print("Successfully converted Laya to Rex checkpoint!")
    total_params = sum(p.numel() for p in rex_model.parameters())
    tool_params = sum(p.numel() for p in rex_model.tool_head.parameters())
    print(f"Total parameters: {total_params:,} (ToolHead: {tool_params:,})")


if __name__ == "__main__":
    main()
