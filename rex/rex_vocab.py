import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn


def check_type_token(tok, token: str) -> int:
    """Validate that a type token is registered and produces exactly one token ID.

    Raises ValueError if the token is not known to the tokenizer or gets split into
    multiple subwords.
    """
    if not isinstance(token, str) or not token:
        raise ValueError(f"Type token must be a non-empty string, got {token!r}")

    # Check that tokenizer recognizes it as a single token
    tokens = tok.tokenize(token)
    if len(tokens) != 1:
        raise ValueError(
            f"Type token {token!r} must tokenize to exactly 1 token, but got {len(tokens)}: {tokens}"
        )
    tid = tok.convert_tokens_to_ids(tokens[0])
    if hasattr(tok, "unk_token_id") and tid == tok.unk_token_id:
        raise ValueError(f"Type token {token!r} resolved to unk_token_id ({tid})")
    return tid


def add_type_tokens(
    tok,
    model: Optional[nn.Module],
    type_tokens: Dict[str, str],
) -> Dict[str, int]:
    """Register special <type> tokens in the tokenizer and initialize their embeddings.

    Parameters
    ----------
    tok : PreTrainedTokenizer
        Tokenizer to add special tokens to.
    model : Optional[nn.Module]
        DecisionModel or RexDecisionModel instance. If provided, embedding weights
        will be resized (if necessary) and the newly added token embeddings will be
        initialized to the mean embedding of their descriptive text.
    type_tokens : Dict[str, str]
        Mapping from token string (e.g. "<TYPE_PROMO_ABUSE>") to a natural language
        description (e.g. "promotion abuse: farming sign-up bonuses and voucher fraud").

    Returns
    -------
    Dict[str, int]
        Mapping from token string to token ID in the expanded tokenizer.
    """
    if not type_tokens:
        return {}

    # 1. Add tokens as special tokens to prevent sub-word splitting and casing changes
    new_tokens = list(type_tokens.keys())
    existing_vocab = tok.get_vocab()
    unseen_tokens = [t for t in new_tokens if t not in existing_vocab]

    if unseen_tokens:
        tok.add_tokens(unseen_tokens, special_tokens=True)

    result_ids = {t: tok.convert_tokens_to_ids(t) for t in new_tokens}

    # 2. If model is provided, resize embeddings and initialize weights
    if model is not None:
        # Locate input embeddings layer across standard architectures and wrappers
        embed_module = None
        encoder = model
        for candidate in (model, getattr(model, "encoder", None)):
            if candidate is None:
                continue
            if hasattr(candidate, "get_input_embeddings") and callable(candidate.get_input_embeddings):
                res = candidate.get_input_embeddings()
                if res is not None and hasattr(res, "weight"):
                    embed_module = res
                    encoder = candidate
                    break
            if hasattr(candidate, "embeddings") and hasattr(candidate.embeddings, "word_embeddings"):
                embed_module = candidate.embeddings.word_embeddings
                encoder = candidate
                break

        if embed_module is None:
            raise AttributeError("Could not locate input embeddings layer on model/encoder.")

        cur_num_embeddings = embed_module.weight.size(0)
        target_vocab_size = len(tok)

        # Only resize if tokenizer length exceeds existing matrix capacity
        # (mmBERT has spare padding rows; BERT/ModernBERT might need resizing)
        if target_vocab_size > cur_num_embeddings:
            if hasattr(encoder, "resize_token_embeddings"):
                encoder.resize_token_embeddings(target_vocab_size)
                if hasattr(encoder, "get_input_embeddings"):
                    embed_module = encoder.get_input_embeddings()
                elif hasattr(encoder, "embeddings") and hasattr(encoder.embeddings, "word_embeddings"):
                    embed_module = encoder.embeddings.word_embeddings
            else:
                raise RuntimeError(
                    f"Encoder has no resize_token_embeddings; cannot resize from {cur_num_embeddings} to {target_vocab_size}"
                )

        # 3. Warm-start embedding with mean of description tokens
        # Only initialize for newly added tokens so that fine-tuned rows are never wiped
        with torch.no_grad():
            weights = embed_module.weight
            for token_str in unseen_tokens:
                desc = type_tokens.get(token_str)
                if not desc or not desc.strip():
                    warnings.warn(
                        f"Empty or missing description for type token {token_str!r}; "
                        f"embedding was left with uncalibrated initialization.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                desc_encoded = tok(desc, add_special_tokens=False)
                desc_ids = desc_encoded.get("input_ids", [])
                if desc_ids:
                    desc_tensor = torch.tensor(desc_ids, device=weights.device, dtype=torch.long)
                    valid_mask = desc_tensor < weights.size(0)
                    if valid_mask.any():
                        mean_vector = weights[desc_tensor[valid_mask]].mean(dim=0)
                        tid = result_ids[token_str]
                        weights[tid].copy_(mean_vector)

    return result_ids
