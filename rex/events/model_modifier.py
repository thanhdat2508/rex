"""Direct model checkpoint modification and vocabulary expansion engine.

Extends the PyTorch decision model's embedding matrix to natively support all <TYPE_...> event
tokens with semantic warm-start weight initialization, saving a deployable checkpoint.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from .types import ALL_EVENT_TOKENS, EventType

logger = logging.getLogger(__name__)


def _extract_token_semantic_words(token_str: str) -> List[str]:
    """Extract constituent semantic words from a token like '<TYPE_CREDENTIAL_STUFFING>'.

    Returns words like ['credential', 'stuffing'].
    """
    cleaned = re.sub(r"^<TYPE_", "", token_str)
    cleaned = re.sub(r">$", "", cleaned)
    words = [w.lower() for w in cleaned.split("_") if w]
    return words


def expand_model_vocabulary(
    model: torch.nn.Module,
    tokenizer: Any,
    new_tokens: Optional[List[str]] = None,
    semantic_init: bool = True,
) -> Dict[str, Any]:
    """Directly modify model embedding weights to incorporate event tokens.

    Args:
        model: DecisionModel instance (e.g. wrapping ModernBERT or other encoder).
        tokenizer: HuggingFace tokenizer.
        new_tokens: List of token strings (defaults to ALL_EVENT_TOKENS).
        semantic_init: If True, initialize new token embeddings as the mean vector
                       of constituent sub-words rather than pure random noise.

    Returns:
        Summary dict of old vs new shapes and initialized token count.
    """
    if new_tokens is None:
        new_tokens = ALL_EVENT_TOKENS

    old_vocab_size = len(tokenizer)
    old_embeddings = model.encoder.get_input_embeddings()
    old_weight = old_embeddings.weight.data.clone()  # Shape: [old_vocab, hidden_dim]
    hidden_dim = old_weight.shape[1]

    # 1. Add new tokens to tokenizer
    existing_special = set(getattr(tokenizer, "all_special_tokens", []))
    tokens_to_add = [tok for tok in new_tokens if tok not in existing_special]
    added_count = tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    new_vocab_size = len(tokenizer)

    # 2. Resize model embeddings in PyTorch
    model.encoder.resize_token_embeddings(new_vocab_size)
    new_embeddings = model.encoder.get_input_embeddings()
    new_weight = new_embeddings.weight.data  # Shape: [new_vocab, hidden_dim]

    # Ensure encoder config reflects the new vocab size
    model.encoder.config.vocab_size = new_vocab_size

    # 3. Semantic Warm-Start Initialization
    initialized_tokens: List[Tuple[str, int]] = []
    if semantic_init and added_count > 0:
        for token_str in tokens_to_add:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if token_id is None or token_id < old_vocab_size:
                continue

            words = _extract_token_semantic_words(token_str)
            sub_token_ids: List[int] = []
            for w in words:
                # Tokenize word using tokens strictly from the old vocabulary
                ids = tokenizer.encode(w, add_special_tokens=False)
                sub_token_ids.extend([i for i in ids if i < old_vocab_size])

            if sub_token_ids:
                # Mean embedding vector of the semantic constituent words
                constituent_vectors = old_weight[sub_token_ids]
                mean_vector = constituent_vectors.mean(dim=0)
                # Blend with subtle gaussian noise to preserve diversity
                noise = torch.randn_like(mean_vector) * (mean_vector.std() * 0.05)
                new_weight[token_id] = mean_vector + noise
            else:
                # Fallback to mean of entire existing vocabulary
                new_weight[token_id] = old_weight.mean(dim=0) + torch.randn(hidden_dim) * 0.02

            initialized_tokens.append((token_str, token_id))

    return {
        "old_vocab_size": old_vocab_size,
        "new_vocab_size": new_vocab_size,
        "added_count": added_count,
        "hidden_dim": hidden_dim,
        "semantic_init": semantic_init,
        "initialized_count": len(initialized_tokens),
        "sample_token_ids": {
            EventType.CREDENTIAL_STUFFING.value: tokenizer.convert_tokens_to_ids(EventType.CREDENTIAL_STUFFING.value),
            EventType.ACCOUNT_TAKEOVER.value: tokenizer.convert_tokens_to_ids(EventType.ACCOUNT_TAKEOVER.value),
            EventType.BENIGN.value: tokenizer.convert_tokens_to_ids(EventType.BENIGN.value),
        },
    }


def save_modified_checkpoint(
    model: torch.nn.Module,
    tokenizer: Any,
    cfg: Dict[str, Any],
    output_path: str,
) -> None:
    """Save the modified model, tokenizer, and config to disk in Laya's standard layout."""
    from safetensors.torch import save_file

    os.makedirs(output_path, exist_ok=True)

    # 1. Save weights in safetensors format (fp16, identical to official Laya checkpoints)
    weights_path = os.path.join(output_path, "model.safetensors")
    save_file(
        {k: v.detach().half().cpu().contiguous() for k, v in model.state_dict().items()},
        weights_path,
    )

    # 2. Save encoder config with updated vocab_size
    encoder_dir = os.path.join(output_path, "encoder")
    os.makedirs(encoder_dir, exist_ok=True)
    model.encoder.config.save_pretrained(encoder_dir)

    # 3. Save expanded tokenizer
    tokenizer_dir = os.path.join(output_path, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_dir)

    # 4. Save updated agent config
    cfg_copy = dict(cfg)
    cfg_copy["event_tokens_version"] = "1.0"
    cfg_copy["event_tokens_count"] = len(ALL_EVENT_TOKENS)
    cfg_copy["vocab_size"] = len(tokenizer)

    tmp_cfg = os.path.join(output_path, "rl_agent_config.json.tmp")
    target_cfg = os.path.join(output_path, "rl_agent_config.json")
    with open(tmp_cfg, "w", encoding="utf-8") as f:
        json.dump(cfg_copy, f, indent=2)
    os.replace(tmp_cfg, target_cfg)

    logger.info("Saved modified checkpoint with %d tokens to %s", len(tokenizer), output_path)


def modify_checkpoint(
    source_model_id_or_path: str,
    output_path: str,
    semantic_init: bool = True,
    verify_load: bool = True,
) -> Dict[str, Any]:
    """End-to-end pipeline: load checkpoint, modify model weights, and save new checkpoint.

    Args:
        source_model_id_or_path: Local path or HF repo ID (e.g. 'convaiinnovations/laya').
        output_path: Destination directory for the new checkpoint.
        semantic_init: Use semantic warm-start embedding initialization.
        verify_load: If True, load the new checkpoint with rex.Agent to verify compatibility.

    Returns:
        Summary dict of modifications.
    """
    from rex.train import load_checkpoint

    logger.info("Loading base checkpoint from %s ...", source_model_id_or_path)
    model, tokenizer, cfg = load_checkpoint(source_model_id_or_path)

    logger.info("Modifying model weights and expanding vocabulary...")
    expansion_stats = expand_model_vocabulary(
        model=model,
        tokenizer=tokenizer,
        new_tokens=ALL_EVENT_TOKENS,
        semantic_init=semantic_init,
    )

    logger.info("Saving modified checkpoint to %s ...", output_path)
    save_modified_checkpoint(model, tokenizer, cfg, output_path)

    verified = False
    if verify_load:
        logger.info("Verifying modified checkpoint with Agent...")
        from rex import Agent
        agent = Agent(output_path, device="cpu")
        # Run a test forward pass using one of the new special tokens
        test_state = f"{EventType.CREDENTIAL_STUFFING.value} 500 failed login attempts from IP 198.51.100.1"
        test_q = {
            "is_threat": {
                "type": "noul",
                "instructions": "Does this represent a threat?",
            }
        }
        res = agent.predict(test_state, test_q)
        answers = res.get("answers", res)
        assert "is_threat" in answers
        verified = True
        logger.info("Checkpoint verified successfully! Answers: %s", answers)

def modify_checkpoint_mece(
    source_model_id_or_path: str,
    output_path: str,
    task_token: str = "<TASK_RISK_INVESTIGATION>",
    verify_load: bool = True,
) -> Dict[str, Any]:
    """End-to-end pipeline: load checkpoint, modify model weights for MECE task token, and save.

    Instead of expanding by 179 tokens, expands by just the unified task token(s), preserving
    maximum model efficiency and stability.
    """
    from rex.train import load_checkpoint
    from .mece import ALL_TASK_TOKENS, format_investigation_state, mece_investigation_question

    logger.info("Loading base checkpoint from %s ...", source_model_id_or_path)
    model, tokenizer, cfg = load_checkpoint(source_model_id_or_path)

    logger.info("Modifying model weights for unified MECE task token: %s", task_token)
    tokens_to_add = list(dict.fromkeys([task_token] + ALL_TASK_TOKENS))

    expansion_stats = expand_model_vocabulary(
        model=model,
        tokenizer=tokenizer,
        new_tokens=tokens_to_add,
        semantic_init=True,
    )

    logger.info("Saving MECE-enabled checkpoint to %s ...", output_path)
    save_modified_checkpoint(model, tokenizer, cfg, output_path)

    verified = False
    if verify_load:
        logger.info("Verifying MECE checkpoint with Agent forward pass...")
        from rex import Agent
        agent = Agent(output_path, device="cpu")

        # Test state with unified task token
        test_state = format_investigation_state({
            "client_ip": "198.51.100.25",
            "user_agent": "python-requests/2.28.1",
            "failed_attempts": 350,
            "path": "/api/v1/auth/login"
        }, task_token=task_token)

        q = mece_investigation_question()
        res = agent.predict(test_state, q)
        answers = res.get("answers", res)
        assert "root_cause" in answers
        verified = True
        logger.info("MECE Checkpoint verified successfully! Probabilities: %s",
                    answers["root_cause"].get("probabilities", answers["root_cause"]))

    return {
        "status": "success",
        "source": source_model_id_or_path,
        "output_path": os.path.abspath(output_path),
        "expansion": expansion_stats,
        "verified": verified,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Modify Laya checkpoint with Event Tokens or MECE Task Token")
    parser.add_argument("--source", default="convaiinnovations/laya", help="Base checkpoint path or repo ID")
    parser.add_argument("--output", default="./checkpoints/laya-mece-investigation", help="Output directory")
    parser.add_argument("--mece", action="store_true", default=True, help="Use single MECE task token (default)")
    parser.add_argument("--all-179-types", action="store_true", help="Expand with all 179 granular types instead")
    parser.add_argument("--no-semantic", action="store_true", help="Disable semantic warm-start initialization")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.all_179_types:
        modify_checkpoint(args.source, args.output, semantic_init=not args.no_semantic)
    else:
        modify_checkpoint_mece(args.source, args.output)
