"""Tokenizer vocabulary expansion and verification for event routing tokens.

Registers all <TYPE_...> tokens as non-splittable special tokens into the tokenizer vocabulary,
enabling System 1 context routing and zero-shot event conditioning.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from .types import ALL_EVENT_TOKENS, EventType

logger = logging.getLogger(__name__)


def register_event_tokens(
    tokenizer: Any,
    model: Optional[Any] = None,
    tokens: Optional[List[str]] = None,
    resize_embeddings: bool = True,
) -> Dict[str, Any]:
    """Register event tokens as additional special tokens in the tokenizer.

    Args:
        tokenizer: Hugging Face PreTrainedTokenizer or PreTrainedTokenizerFast.
        model: Optional PyTorch model whose embedding matrix should be resized.
        tokens: Custom token list, defaults to ALL_EVENT_TOKENS (~179 tokens).
        resize_embeddings: If True and model is provided, resize token embeddings.

    Returns:
        Summary dict containing added_count, vocab_size, and verification status.
    """
    if tokens is None:
        tokens = ALL_EVENT_TOKENS

    initial_vocab_size = len(tokenizer)

    # Filter out tokens that are already present as exact single tokens
    existing_special: Set[str] = set(getattr(tokenizer, "all_special_tokens", []))
    tokens_to_add = [tok for tok in tokens if tok not in existing_special]

    added_count = 0
    if tokens_to_add:
        # Register as additional_special_tokens so they are never split by the tokenizer
        added_count = tokenizer.add_special_tokens({
            "additional_special_tokens": tokens_to_add
        })
        logger.info("Added %d new special event tokens to tokenizer (requested %d)", added_count, len(tokens_to_add))

    # Resize model embeddings if model is provided
    model_resized = False
    if model is not None and resize_embeddings and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
        model_resized = True

    # Verification check
    verification = verify_event_tokens(tokenizer, tokens)

    return {
        "initial_vocab_size": initial_vocab_size,
        "final_vocab_size": len(tokenizer),
        "added_count": added_count,
        "model_resized": model_resized,
        "all_valid": verification["all_valid"],
        "invalid_count": len(verification["invalid_tokens"]),
        "sample_token_ids": {
            EventType.CREDENTIAL_STUFFING.value: tokenizer.convert_tokens_to_ids(EventType.CREDENTIAL_STUFFING.value),
            EventType.ACCOUNT_TAKEOVER.value: tokenizer.convert_tokens_to_ids(EventType.ACCOUNT_TAKEOVER.value),
            EventType.BENIGN.value: tokenizer.convert_tokens_to_ids(EventType.BENIGN.value),
        },
    }


def verify_event_tokens(
    tokenizer: Any,
    tokens: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Verify that all event tokens are encoded as exactly ONE token ID (not split).

    Args:
        tokenizer: Tokenizer to verify.
        tokens: Tokens to check (defaults to ALL_EVENT_TOKENS).

    Returns:
        Dict with 'all_valid' boolean and list of any invalid/split tokens.
    """
    if tokens is None:
        tokens = ALL_EVENT_TOKENS

    invalid_tokens: List[Tuple[str, List[int]]] = []
    unk_token_id = getattr(tokenizer, "unk_token_id", None)

    for tok in tokens:
        encoded = tokenizer.encode(tok, add_special_tokens=False)
        # Must be exactly 1 token and not mapped to <unk> if unk exists
        if len(encoded) != 1 or (unk_token_id is not None and encoded[0] == unk_token_id):
            invalid_tokens.append((tok, encoded))

    return {
        "all_valid": len(invalid_tokens) == 0,
        "checked_count": len(tokens),
        "valid_count": len(tokens) - len(invalid_tokens),
        "invalid_tokens": invalid_tokens,
    }


def get_token_id(tokenizer: Any, event_type: EventType) -> Optional[int]:
    """Get the token ID for a specific EventType."""
    token_str = event_type.value if isinstance(event_type, EventType) else event_type
    token_id = tokenizer.convert_tokens_to_ids(token_str)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if token_id == unk_id and token_str != getattr(tokenizer, "unk_token", None):
        return None
    return token_id


def register_task_token(
    tokenizer: Any,
    model: Optional[Any] = None,
    task_token: str = "<TASK_RISK_INVESTIGATION>",
    resize_embeddings: bool = True,
) -> Dict[str, Any]:
    """Register a single unified task token into the tokenizer vocabulary."""
    from .mece import ALL_TASK_TOKENS

    tokens_to_add = [t for t in ([task_token] + ALL_TASK_TOKENS) if t not in getattr(tokenizer, "all_special_tokens", [])]
    # Remove duplicates preserving order
    unique_tokens = list(dict.fromkeys(tokens_to_add))

    initial_vocab_size = len(tokenizer)
    added_count = 0
    if unique_tokens:
        added_count = tokenizer.add_special_tokens({"additional_special_tokens": unique_tokens})

    model_resized = False
    if model is not None and resize_embeddings and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
        model_resized = True

    # Verify single token encoding
    token_id = tokenizer.convert_tokens_to_ids(task_token)
    encoded = tokenizer.encode(task_token, add_special_tokens=False)
    is_valid = len(encoded) == 1 and encoded[0] == token_id

    return {
        "initial_vocab_size": initial_vocab_size,
        "final_vocab_size": len(tokenizer),
        "added_count": added_count,
        "task_token": task_token,
        "token_id": token_id,
        "is_valid": is_valid,
        "model_resized": model_resized,
    }


def verify_mece_hypotheses_lengths(
    tokenizer: Any,
    min_tokens: int = 10,
    max_tokens: int = 25,
) -> Dict[str, Any]:
    """Verify that each of the 4 MECE hypothesis descriptions satisfies the 10-25 token constraint."""
    from .mece import MECE_HYPOTHESIS_STRINGS

    lengths: Dict[str, int] = {}
    tokens_map: Dict[str, List[int]] = {}
    violations: List[str] = []

    for hyp, text in MECE_HYPOTHESIS_STRINGS.items():
        encoded = tokenizer.encode(text, add_special_tokens=False)
        tok_len = len(encoded)
        lengths[hyp.value] = tok_len
        tokens_map[hyp.value] = encoded
        if not (min_tokens <= tok_len <= max_tokens):
            violations.append(f"{hyp.value}: length={tok_len} outside [{min_tokens}, {max_tokens}]")

    return {
        "all_valid": len(violations) == 0,
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "lengths": lengths,
        "violations": violations,
    }


def setup_investigation_tokenizer(
    tokenizer: Any,
    model: Optional[Any] = None,
    domain_tokens: Optional[List[str]] = None,
    mask_token: str = "[MASK]",
    cls_token: str = "[CLS]",
    sep_token: str = "[SEP]",
    resize_embeddings: bool = True,
) -> Dict[str, Any]:
    """Configure and verify tokenizer for Laya Investigation & Falsification architecture.

    Guarantees:
      1. Domain tokens (<TYPE_PROMO_ABUSE>, <TYPE_CREDENTIAL_STUFFING>, etc.) are atomic single tokens.
      2. The [MASK] token is present in the vocabulary and configured as tokenizer.mask_token.
      3. [CLS] and [SEP] tokens are present and accessible.
      4. Model embedding matrix is resized if a model instance is provided.

    Returns:
        Summary dict containing vocab changes, token IDs, and validation status.
    """
    from .mece import ALL_TASK_TOKENS

    initial_vocab_size = len(tokenizer)
    tokens_to_add: List[str] = []

    # 1. Domain Event & Task Tokens
    if domain_tokens is None:
        target_tokens = ALL_EVENT_TOKENS + ALL_TASK_TOKENS
    else:
        target_tokens = domain_tokens

    existing_special = set(getattr(tokenizer, "all_special_tokens", []))
    for tok in target_tokens:
        if tok not in existing_special and tok not in tokenizer.get_vocab():
            tokens_to_add.append(tok)

    # 2. Ensure [MASK] exists
    mask_in_vocab = mask_token in tokenizer.get_vocab() or mask_token in existing_special
    if not mask_in_vocab:
        tokenizer.add_special_tokens({"mask_token": mask_token})
        existing_special.add(mask_token)
    elif getattr(tokenizer, "mask_token", None) is None:
        tokenizer.mask_token = mask_token
        tokenizer.mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)

    # 3. Ensure [CLS] and [SEP] exist
    if cls_token not in tokenizer.get_vocab() and cls_token not in existing_special:
        tokenizer.add_special_tokens({"cls_token": cls_token})
        existing_special.add(cls_token)
    elif getattr(tokenizer, "cls_token", None) is None:
        tokenizer.cls_token = cls_token
        tokenizer.cls_token_id = tokenizer.convert_tokens_to_ids(cls_token)

    if sep_token not in tokenizer.get_vocab() and sep_token not in existing_special:
        tokenizer.add_special_tokens({"sep_token": sep_token})
        existing_special.add(sep_token)
    elif getattr(tokenizer, "sep_token", None) is None:
        tokenizer.sep_token = sep_token
        tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)

    # 4. Add additional special tokens
    added_count = 0
    if tokens_to_add:
        added_count = tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        logger.info("Registered %d domain & task tokens into tokenizer", added_count)

    # 5. Resize model embeddings if requested
    model_resized = False
    if model is not None and resize_embeddings and hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
        model_resized = True

    # 6. Verify key tokens
    mask_token_id = tokenizer.convert_tokens_to_ids(mask_token)
    cls_token_id = tokenizer.convert_tokens_to_ids(cls_token)
    sep_token_id = tokenizer.convert_tokens_to_ids(sep_token)

    promo_abuse_id = tokenizer.convert_tokens_to_ids(EventType.PROMO_ABUSE.value)
    cred_stuffing_id = tokenizer.convert_tokens_to_ids(EventType.CREDENTIAL_STUFFING.value)

    # Check that they encode to exactly 1 token ID
    def is_atomic(t: str) -> bool:
        enc = tokenizer.encode(t, add_special_tokens=False)
        return len(enc) == 1

    all_atomic = (
        is_atomic(mask_token)
        and is_atomic(cls_token)
        and is_atomic(sep_token)
        and is_atomic(EventType.PROMO_ABUSE.value)
        and is_atomic(EventType.CREDENTIAL_STUFFING.value)
    )

    return {
        "initial_vocab_size": initial_vocab_size,
        "final_vocab_size": len(tokenizer),
        "added_count": added_count,
        "mask_token": mask_token,
        "mask_token_id": mask_token_id,
        "cls_token": cls_token,
        "cls_token_id": cls_token_id,
        "sep_token": sep_token,
        "sep_token_id": sep_token_id,
        "promo_abuse_id": promo_abuse_id,
        "cred_stuffing_id": cred_stuffing_id,
        "model_resized": model_resized,
        "all_atomic": all_atomic,
    }

