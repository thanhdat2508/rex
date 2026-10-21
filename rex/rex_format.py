"""Input layout formatting and batch collation for Rex."""
from typing import Any, Dict, List, Optional, Sequence, Union

import torch

from .common import build_sequence
from .rex_vocab import check_type_token


def encode_rex_input(
    tok,
    type_token: str,
    instruction: str,
    hypotheses: Dict[str, str],
    state: Union[str, dict, list] = "",
    alive: Optional[Dict[str, bool]] = None,
    max_len: int = 512,
    head_max_len: int = 192,
    truncate_left: bool = True,
    option_order: Optional[List[int]] = None,
    tool_mask: Optional[List[bool]] = None,
) -> Dict[str, Any]:
    """Encode an incident investigation query into Rex input tokens with validation.

    Layout:
        [CLS] choice question: <type> instructions [SEP] [MASK] H_0 ... [MASK] H_novel [SEP] state_t [SEP]

    Parameters
    ----------
    tok : PreTrainedTokenizer
        Tokenizer configured with special type tokens.
    type_token : str
        Domain event identifier token (e.g. "<TYPE_PROMO_ABUSE>").
    instruction : str
        Investigation prompt / hypothesis question.
    hypotheses : Dict[str, str]
        Ordered mapping of hypothesis name -> descriptive text.
        Must end with "H_novel".
    state : Union[str, dict, list]
        Accumulated evidence observation state.
    alive : Optional[Dict[str, bool]]
        Survival mask for Popperian elimination. "H_novel" cannot be marked False.
    max_len : int
        Maximum total sequence length (default 512).
    head_max_len : int
        Maximum length reserved for question and hypothesis header (default 192).
    truncate_left : bool
        If True, truncate earlier evidence in state when budget is exceeded.
    option_order : Optional[List[int]]
        Explicit permutation of hypotheses to mitigate position bias.
    tool_mask : Optional[List[bool]]
        Optional boolean mask for available tools.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing input_ids, markers, hyp_keys, hyp_alive, truncation stats, etc.
    """
    # 1. Validate type token
    type_tid = check_type_token(tok, type_token)

    # 2. Validate hypotheses structure
    if not isinstance(hypotheses, dict) or len(hypotheses) < 2:
        raise ValueError("hypotheses must be a dictionary with at least 2 hypotheses (including H_novel)")

    hyp_keys_list = list(hypotheses.keys())
    if hyp_keys_list[-1] != "H_novel":
        raise ValueError(
            f"The final hypothesis slot must be 'H_novel', but found '{hyp_keys_list[-1]}'"
        )

    alive_map = alive or {}
    unknown_alive = set(alive_map.keys()) - set(hypotheses.keys())
    if unknown_alive:
        raise ValueError(f"Unknown hypothesis keys in alive mask: {sorted(unknown_alive)}")
    if alive_map.get("H_novel") is False:
        raise ValueError("Slot 'H_novel' cannot be eliminated; it represents the unexplained residual.")

    # 3. Assemble sequence using Laya's build_sequence
    ins_text = f"{type_token} {instruction.strip()}"
    q = {
        "t": "choice",
        "ins": ins_text,
        "crit": hypotheses,
    }

    ids, markers, stats, trunc_stats = build_sequence(
        tok,
        state=state,
        q=q,
        max_len=max_len,
        head_max_len=head_max_len,
        truncate_left=truncate_left,
        option_order=option_order,
        return_truncation_stats=True,
        return_stats=True,
    )

    # 4. Strict validations against truncation of critical components
    if len(markers) != len(hypotheses):
        raise ValueError(
            f"Expected {len(hypotheses)} hypothesis markers, but only {len(markers)} survived truncation. "
            f"Increase max_len or head_max_len."
        )

    if stats.get("options_distinct", len(hypotheses)) < stats.get("options", len(hypotheses)):
        raise ValueError("Multiple hypotheses truncated to identical token sequences.")

    if type_tid not in ids:
        raise ValueError(f"Type token {type_token} was truncated from sequence header.")

    # 5. Map hypothesis keys and alive flags according to effective option order
    effective_order = option_order if option_order is not None else list(range(len(hyp_keys_list)))
    ordered_keys = [hyp_keys_list[i] for i in effective_order]
    ordered_alive = [alive_map.get(k, True) for k in ordered_keys]

    return {
        "input_ids": ids,
        "markers": markers,
        "hyp_keys": ordered_keys,
        "hyp_alive": ordered_alive,
        "truncation": trunc_stats,
        "tool_mask": tool_mask,
    }


def collate_rex(
    batch: Sequence[Dict[str, Any]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    """Collate a list of encode_rex_input outputs into a padded batch for RexDecisionModel.

    Parameters
    ----------
    batch : Sequence[Dict[str, Any]]
        List of item dictionaries produced by encode_rex_input.
    pad_token_id : int
        Tokenizer padding token ID.

    Returns
    -------
    Dict[str, torch.Tensor]
        Batch ready for model.rex_forward:
        - input_ids: [B, max_len]
        - attention_mask: [B, max_len]
        - marker_pos: [B, max_markers]
        - marker_mask: [B, max_markers] bool
        - hyp_alive: [B, max_markers] bool
        - qtype: [B]
        - tool_mask: Optional [B, n_tools] bool
    """
    b = len(batch)
    max_len = max(len(item["input_ids"]) for item in batch)
    max_markers = max(len(item["markers"]) for item in batch)

    input_ids = torch.full((b, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((b, max_len), dtype=torch.long)
    marker_pos = torch.zeros((b, max_markers), dtype=torch.long)
    marker_mask = torch.zeros((b, max_markers), dtype=torch.bool)
    hyp_alive = torch.zeros((b, max_markers), dtype=torch.bool)

    has_tool_mask = any(item.get("tool_mask") is not None for item in batch)
    tool_mask_tensor = None
    if has_tool_mask:
        n_tools = max(len(item["tool_mask"]) for item in batch if item.get("tool_mask") is not None)
        tool_mask_tensor = torch.ones((b, n_tools), dtype=torch.bool)

    for i, item in enumerate(batch):
        ids = item["input_ids"]
        seq_len = len(ids)
        input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, :seq_len] = 1

        markers = item["markers"]
        m_len = len(markers)
        marker_pos[i, :m_len] = torch.tensor(markers, dtype=torch.long)
        marker_mask[i, :m_len] = True

        alive_flags = item.get("hyp_alive", [True] * m_len)
        hyp_alive[i, :m_len] = torch.tensor(alive_flags, dtype=torch.bool)

        if tool_mask_tensor is not None and item.get("tool_mask") is not None:
            tm = item["tool_mask"]
            tool_mask_tensor[i, : len(tm)] = torch.tensor(tm, dtype=torch.bool)

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "hyp_alive": hyp_alive,
        "qtype": torch.zeros(b, dtype=torch.long),
    }
    return result


def encode_evidence(
    tok: Any,
    evidence_text: Union[str, Dict[str, Any], List[Any]],
    round_idx: int = 1,
    max_tokens: int = 64,
) -> Dict[str, Any]:
    """Encode an incremental evidence piece with a strict token budget cap (Step A4).

    Parameters
    ----------
    tok : Any
        Tokenizer configured for the model.
    evidence_text : Union[str, Dict[str, Any], List[Any]]
        Observation string or structured dictionary from tools.
    round_idx : int
        Investigation turn index (default: 1).
    max_tokens : int
        Hard upper bound on encoded evidence tokens (default: 64).

    Returns
    -------
    Dict[str, Any]
        {
            "input_ids": list of token IDs,
            "attention_mask": list of binary mask flags,
            "is_truncated": bool flag,
            "round_idx": int,
        }
    """
    import json

    if not isinstance(evidence_text, str):
        text = json.dumps(evidence_text, ensure_ascii=False)
    else:
        text = evidence_text.strip()

    if hasattr(tok, "encode"):
        ids = tok.encode(text, add_special_tokens=False)
    else:
        # Fallback dummy encoding for token-free mocking in unit tests
        ids = [int(abs(hash(w)) % 1000) + 1 for w in text.split()] or [1]

    is_truncated = len(ids) > max_tokens
    if is_truncated:
        ids = ids[:max_tokens]

    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "is_truncated": is_truncated,
        "round_idx": round_idx,
    }
