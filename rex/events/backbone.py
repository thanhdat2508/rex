"""Backbone Refactoring: Tensor Formatting, Position Tracker, Tool MLP Head, and Shared Verdict Head.

Implements Step 2 of the Falsification & MECE Architecture:
  1. Standardized input sequence formatting:
     [CLS] <type> instructions [SEP] [MASK] H_0 ... [MASK] H_novel [SEP] state_t [SEP]
  2. PositionTracker: extracts exact indices of [CLS] and all [MASK]_j tokens.
  3. ToolMLPHead: 2-layer MLP with intermediate LayerNorm & GELU projecting h_CLS to Tool Slot logits.
  4. SharedVerdictHead: Single shared linear projection layer scoring hypothesis confidence logits at each [MASK]_j.
  5. LayaInvestigationModel: Unified PyTorch module connecting backbone with PoE Inverse Tool Masking & Budget Pruner K_t.
"""

from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .falsification import (
    SAFE_FP16_MASK_VALUE,
    HypothesisMask,
    ToolRegistry,
    ToolSlot,
    apply_inverse_tool_logits_mask,
    prune_slots_by_budget,
)
from .mece import DEFAULT_TASK_TOKEN, MECE_HYPOTHESIS_STRINGS, RiskHypothesis
from .types import EventType


# Canonical order of MECE Hypotheses for tensor formatting and indexing
CANONICAL_HYPOTHESES: List[RiskHypothesis] = [
    RiskHypothesis.BOTNET,
    RiskHypothesis.SYBIL,
    RiskHypothesis.SHARED,
    RiskHypothesis.NOVEL,
]

CANONICAL_TOOL_SLOTS: List[ToolSlot] = [
    ToolSlot.NETWORK,
    ToolSlot.DEVICE,
    ToolSlot.BEHAVIOR,
    ToolSlot.GRAPH,
]


def compact_telemetry_string(payload: Union[Dict[str, Any], str, List[Any]], max_val_len: int = 32) -> str:
    """Format telemetry payload into concise, token-efficient key-values (omitting JSON syntax noise)."""
    if not isinstance(payload, dict):
        if isinstance(payload, list):
            return " ".join(compact_telemetry_string(x, max_val_len) for x in payload)
        return str(payload).strip()

    parts: List[str] = []
    skip_keys = {"raw_payload", "payload_summary", "headers", "stack_trace", "jwt_raw"}
    for k, v in payload.items():
        if k in skip_keys or v is None:
            continue
        if isinstance(v, float):
            val_str = f"{v:.2f}"
        elif isinstance(v, bool):
            val_str = "1" if v else "0"
        elif isinstance(v, dict):
            val_str = compact_telemetry_string(v, max_val_len)
        else:
            val_str = str(v).strip()
            if len(val_str) > max_val_len:
                val_str = val_str[:max_val_len] + ".."
        parts.append(f"{k}={val_str}")
    return " ".join(parts)


class InvestigationTensorFormatter:
    """Formats raw investigation inputs into the standardized token sequence:

    [CLS] <type> instructions [SEP] [MASK] H_0 ... [MASK] H_novel [SEP] state_t [SEP]
    """

    def __init__(
        self,
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        mask_token: str = "[MASK]",
    ):
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token

    def format_sequence(
        self,
        type_token: Union[str, EventType] = DEFAULT_TASK_TOKEN,
        instructions: str = "Identify root cause hypothesis and route investigation",
        state_payload: Union[Dict[str, Any], str] = "",
        hypotheses: Optional[Dict[RiskHypothesis, str]] = None,
        tokenizer: Optional[Any] = None,
        max_length: Optional[int] = None,
        truncate_left: bool = True,
        compact_state: bool = False,
    ) -> str:
        """Format a single investigation example into the standardized token string:

        [CLS] <type> instructions [SEP] [MASK] H_0 ... [MASK] H_novel [SEP] state_t [SEP]

        Guarantees that prefix containing all 4 [MASK] tokens and separating [SEP] tokens
        is NEVER truncated. Truncation strictly applies only to state_t.
        """
        # Type token string
        type_str = type_token.value if isinstance(type_token, EventType) else str(type_token).strip()

        # Instruction string
        inst_str = instructions.strip()

        # MECE Hypotheses segment with [MASK] prefixes
        hyp_dict = hypotheses or MECE_HYPOTHESIS_STRINGS
        hyp_parts: List[str] = []
        for hyp in CANONICAL_HYPOTHESES:
            desc = hyp_dict.get(hyp, "").strip()
            hyp_parts.append(f"{self.mask_token} {desc}")
        hypotheses_str = " ".join(hyp_parts)

        # Invariant Prefix
        prefix = f"{self.cls_token} {type_str} {inst_str} {self.sep_token} {hypotheses_str} {self.sep_token}"

        # State telemetry string
        if isinstance(state_payload, (dict, list)):
            if compact_state:
                state_str = compact_telemetry_string(state_payload)
            else:
                state_str = json.dumps(state_payload, ensure_ascii=False)
        else:
            state_str = str(state_payload).strip()

        # Apply state_t-only truncation with truncate_left (preserving freshest evidence)
        if tokenizer is not None and max_length is not None:
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            sep_ids = tokenizer.encode(self.sep_token, add_special_tokens=False)
            reserved = len(prefix_ids) + len(sep_ids) + 2
            avail_state_tokens = max(0, max_length - reserved)

            state_ids = tokenizer.encode(state_str, add_special_tokens=False)
            if len(state_ids) > avail_state_tokens:
                # Left-truncation: discard oldest telemetry, strictly preserve freshest round observations
                truncated_state_ids = state_ids[-avail_state_tokens:] if avail_state_tokens > 0 else []
                state_str = tokenizer.decode(truncated_state_ids, skip_special_tokens=True).strip()

        # Assemble sequence: Invariant prefix is guaranteed 100% intact
        return f"{prefix} {state_str} {self.sep_token}"

    def encode_batch(
        self,
        tokenizer: Any,
        items: List[Dict[str, Any]],
        max_length: int = 512,
        padding: bool = True,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Format and tokenize a batch of items with guaranteed state_t-only truncation.

        The Invariant Prefix:
            [CLS] <type> instructions [SEP] [MASK] H_0 ... [MASK] H_novel [SEP]
        and trailing [SEP] are 100% protected from truncation. When sequence exceeds max_length,
        only the interior telemetry tokens in state_t are sliced.

        Returns:
            Dict containing input_ids, attention_mask, cls_indices, and mask_indices.
        """
        sep_token_id = getattr(tokenizer, "sep_token_id", None) or tokenizer.convert_tokens_to_ids(self.sep_token)
        mask_token_id = getattr(tokenizer, "mask_token_id", None) or tokenizer.convert_tokens_to_ids(self.mask_token)
        cls_token_id = getattr(tokenizer, "cls_token_id", None) or tokenizer.convert_tokens_to_ids(self.cls_token)

        batch_token_ids: List[List[int]] = []

        for item in items:
            type_token = item.get("type_token", DEFAULT_TASK_TOKEN)
            instructions = item.get("instructions", "Identify root cause hypothesis and route investigation")
            state_payload = item.get("state_payload", {})
            hypotheses = item.get("hypotheses", None)

            # Type token string
            type_str = type_token.value if isinstance(type_token, EventType) else str(type_token).strip()
            inst_str = instructions.strip()

            # MECE Hypotheses segment
            hyp_dict = hypotheses or MECE_HYPOTHESIS_STRINGS
            hyp_parts = [f"{self.mask_token} {hyp_dict.get(h, '').strip()}" for h in CANONICAL_HYPOTHESES]
            hypotheses_str = " ".join(hyp_parts)

            # Tokenize prefix
            prefix = f"{self.cls_token} {type_str} {inst_str} {self.sep_token} {hypotheses_str} {self.sep_token}"
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)

            # Tokenize state
            if isinstance(state_payload, (dict, list)):
                state_str = json.dumps(state_payload, ensure_ascii=False)
            else:
                state_str = str(state_payload).strip()
            state_ids = tokenizer.encode(state_str, add_special_tokens=False)

            # Suffix [SEP]
            suffix_ids = [sep_token_id]

            # Strict state_t-only truncation
            reserved = len(prefix_ids) + len(suffix_ids)
            avail_state_tokens = max(0, max_length - reserved)
            truncated_state_ids = state_ids[:avail_state_tokens]

            seq_ids = prefix_ids + truncated_state_ids + suffix_ids
            batch_token_ids.append(seq_ids)

        # Apply padding across batch
        max_batch_len = max(len(s) for s in batch_token_ids) if batch_token_ids else 0
        pad_token_id = getattr(tokenizer, "pad_token_id", 0) or 0

        input_ids_list: List[List[int]] = []
        attention_mask_list: List[List[int]] = []

        for s in batch_token_ids:
            if padding:
                pad_len = max_batch_len - len(s)
                input_ids_list.append(s + [pad_token_id] * pad_len)
                attention_mask_list.append([1] * len(s) + [0] * pad_len)
            else:
                input_ids_list.append(s)
                attention_mask_list.append([1] * len(s))

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)

        # Extract positions
        cls_indices, mask_indices = PositionTracker.extract_positions(
            input_ids=input_ids,
            mask_token_id=mask_token_id,
            cls_token_id=cls_token_id,
            expected_num_masks=len(CANONICAL_HYPOTHESES),
        )

        batch = {
            "input_ids": input_ids.to(device) if device else input_ids,
            "attention_mask": attention_mask.to(device) if device else attention_mask,
            "cls_indices": cls_indices.to(device) if device else cls_indices,
            "mask_indices": mask_indices.to(device) if device else mask_indices,
        }
        return batch



class PositionTracker:
    """Robust extractor and validator for token positions ([CLS] and [MASK]_j) in token tensors."""

    @staticmethod
    def extract_positions(
        input_ids: torch.Tensor,
        mask_token_id: int,
        cls_token_id: Optional[int] = None,
        expected_num_masks: int = 4,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract exact index of [CLS] and all [MASK]_j tokens per sequence.

        Args:
            input_ids: Tensor of shape [batch_size, seq_len].
            mask_token_id: Token ID representing [MASK].
            cls_token_id: Token ID representing [CLS].
            expected_num_masks: Number of hypotheses (default: 4 for MECE).

        Returns:
            Tuple of:
              - cls_indices: Tensor of shape [batch_size] containing CLS token index per row.
              - mask_indices: Tensor of shape [batch_size, expected_num_masks] containing [MASK] indices.

        Raises:
            ValueError: If any sequence does not contain exactly expected_num_masks [MASK] tokens.
        """
        batch_size = input_ids.size(0)
        cls_indices_list: List[int] = []
        mask_indices_list: List[torch.Tensor] = []

        for b in range(batch_size):
            row = input_ids[b]

            # 1. Locate [CLS] token
            if cls_token_id is not None:
                matches_cls = (row == cls_token_id).nonzero(as_tuple=True)[0]
                if len(matches_cls) > 0:
                    cls_idx = int(matches_cls[0].item())
                else:
                    cls_idx = 0  # Fallback to index 0
            else:
                cls_idx = 0
            cls_indices_list.append(cls_idx)

            # 2. Locate all [MASK] tokens
            matches_mask = (row == mask_token_id).nonzero(as_tuple=True)[0]
            if len(matches_mask) != expected_num_masks:
                raise ValueError(
                    f"Batch item {b} has {len(matches_mask)} [MASK] tokens (ID={mask_token_id}), "
                    f"expected exactly {expected_num_masks}. Check truncation or formatting!"
                )
            mask_indices_list.append(matches_mask)

        cls_indices = torch.tensor(cls_indices_list, dtype=torch.long, device=input_ids.device)
        mask_indices = torch.stack(mask_indices_list, dim=0).to(input_ids.device)

        return cls_indices, mask_indices

    @staticmethod
    def extract_spans(
        input_ids: torch.Tensor,
        mask_indices: torch.Tensor,
        sep_token_id: Optional[int] = None,
    ) -> Tuple[List[List[Tuple[int, int]]], List[Tuple[int, int]]]:
        """Extract [start, end] inclusive token index ranges for each hypothesis and state_t.

        Sequence Layout:
          [CLS] <type> instructions [SEP] [MASK]_0 H_0 ... [MASK]_3 H_novel [SEP] state_t [SEP]

        Returns:
            Tuple of:
              - all_hyp_spans: List of length batch_size, where each element is a list of 4 (start, end)
                               tuples representing the token span for [H_botnet, H_sybil, H_shared, H_novel].
              - all_state_spans: List of length batch_size containing (state_start, state_end) tuples.
        """
        batch_size = input_ids.size(0)
        num_hypotheses = mask_indices.size(1)
        all_hyp_spans: List[List[Tuple[int, int]]] = []
        all_state_spans: List[Tuple[int, int]] = []

        for b in range(batch_size):
            row = input_ids[b]
            masks = mask_indices[b]

            # Find all SEP tokens in this row
            if sep_token_id is not None:
                sep_matches = (row == sep_token_id).nonzero(as_tuple=True)[0]
            else:
                sep_matches = torch.tensor([], dtype=torch.long)

            # Hypothesis 0..2 spans: from masks[j] up to masks[j+1] - 1
            hyp_spans: List[Tuple[int, int]] = []
            for j in range(num_hypotheses - 1):
                s = int(masks[j].item())
                e = int(masks[j + 1].item()) - 1
                hyp_spans.append((s, max(s, e)))

            # For the last hypothesis (H_novel):
            last_mask = int(masks[-1].item())
            seps_after_mask = sep_matches[sep_matches > last_mask] if len(sep_matches) > 0 else []

            if len(seps_after_mask) > 0:
                intermediate_sep = int(seps_after_mask[0].item())
                hyp_spans.append((last_mask, max(last_mask, intermediate_sep - 1)))
                state_start = intermediate_sep + 1
                if len(seps_after_mask) > 1:
                    state_end = int(seps_after_mask[-1].item()) - 1
                else:
                    state_end = int(row.size(0) - 1)
            else:
                hyp_spans.append((last_mask, int(row.size(0) - 1)))
                state_start = int(row.size(0) - 1)
                state_end = int(row.size(0) - 1)

            all_hyp_spans.append(hyp_spans)
            all_state_spans.append((state_start, max(state_start, state_end)))

        return all_hyp_spans, all_state_spans


def build_poe_attention_bias(
    input_ids: torch.Tensor,
    mask_indices: torch.Tensor,
    hypothesis_mask: Union[HypothesisMask, torch.Tensor, List[int]],
    cls_indices: Optional[torch.Tensor] = None,
    base_attention_mask: Optional[torch.Tensor] = None,
    sep_token_id: Optional[int] = None,
    mask_value: float = SAFE_FP16_MASK_VALUE,
) -> torch.Tensor:
    """Construct 4D additive attention bias tensor [B, 1, S, S] enforcing Process of Elimination (PoE).

    Architectural Injection into Transformer Self-Attention:
      Attention(Q, K, V) = softmax((QK^T / sqrt(d_k)) + Bias) * V

    When hypothesis H_j is falsified (M_j = 0):
      1. Blocks bidirectional attention between [CLS] and tokens in H_j span:
         h_CLS cannot attend to or be computed from refuted hypotheses.
      2. Blocks bidirectional attention between state_t and tokens in H_j span:
         refuted hypotheses cannot absorb observation evidence, and state_t representations are not distracted.
      3. Blocks cross-attention between surviving active hypotheses and refuted hypotheses.
      4. Preserves full attention channels for active hypotheses (M_j = 1) and residual H_novel (M_novel ≡ 1).

    Returns:
        4D additive tensor [batch_size, 1, seq_len, seq_len] safe for FP16 and BF16.
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Initialize 4D attention bias tensor [B, 1, S, S] with 0.0 (unmasked)
    bias = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=torch.float32, device=device)

    # Incorporate base attention mask (e.g. padding tokens)
    if base_attention_mask is not None:
        pad_mask = (base_attention_mask == 0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
        bias = bias.masked_fill(pad_mask, mask_value)

    # Extract exact spans
    all_hyp_spans, all_state_spans = PositionTracker.extract_spans(
        input_ids=input_ids,
        mask_indices=mask_indices,
        sep_token_id=sep_token_id,
    )

    # Normalize hypothesis_mask to tensor [B, 4]
    if isinstance(hypothesis_mask, HypothesisMask):
        m_list = hypothesis_mask.to_list()
        m_tensor = torch.tensor([m_list] * batch_size, dtype=torch.long, device=device)
    elif isinstance(hypothesis_mask, (list, tuple)):
        m_tensor = torch.tensor([list(hypothesis_mask)] * batch_size, dtype=torch.long, device=device)
    elif isinstance(hypothesis_mask, torch.Tensor):
        if hypothesis_mask.dim() == 1:
            m_tensor = hypothesis_mask.unsqueeze(0).expand(batch_size, -1).to(device)
        else:
            m_tensor = hypothesis_mask.to(device)
    else:
        m_tensor = torch.ones((batch_size, 4), dtype=torch.long, device=device)

    for b in range(batch_size):
        cls_idx = int(cls_indices[b].item()) if cls_indices is not None else 0
        state_start, state_end = all_state_spans[b]
        hyp_spans = all_hyp_spans[b]

        for j in range(len(hyp_spans)):
            # Check if hypothesis j is falsified (M_j == 0)
            if m_tensor[b, j].item() == 0:
                h_start, h_end = hyp_spans[j]

                # 1. Block [CLS] <-> Refuted Hypothesis
                bias[b, 0, cls_idx, h_start : h_end + 1] = mask_value
                bias[b, 0, h_start : h_end + 1, cls_idx] = mask_value

                # 2. Block state_t <-> Refuted Hypothesis
                if state_start <= state_end and state_end < seq_len:
                    bias[b, 0, state_start : state_end + 1, h_start : h_end + 1] = mask_value
                    bias[b, 0, h_start : h_end + 1, state_start : state_end + 1] = mask_value

                # 3. Block active hypotheses <-> Refuted Hypothesis
                for k in range(len(hyp_spans)):
                    if k != j and m_tensor[b, k].item() == 1:
                        k_start, k_end = hyp_spans[k]
                        bias[b, 0, k_start : k_end + 1, h_start : h_end + 1] = mask_value
                        bias[b, 0, h_start : h_end + 1, k_start : k_end + 1] = mask_value

    return bias


# 3. TOOL MLP HEAD

class ToolMLPHead(nn.Module):
    """2-layer MLP with intermediate LayerNorm & GELU projecting h_CLS to Tool Slot logits.

    Architecture:
        h_CLS [B, D] -> Linear(D, intermediate_size) -> LayerNorm -> GELU -> Dropout -> Linear(intermediate_size, num_slots)
    """

    def __init__(
        self,
        hidden_size: int,
        num_slots: int = 4,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        intermediate_size = intermediate_size or hidden_size
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.ln = nn.LayerNorm(intermediate_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(intermediate_size, num_slots)

    def forward(self, h_cls: torch.Tensor) -> torch.Tensor:
        """Project h_CLS hidden state into raw Tool Slot logits.

        Args:
            h_cls: Tensor of shape [batch_size, hidden_size].

        Returns:
            tool_logits: Tensor of shape [batch_size, num_slots].
        """
        x = self.fc1(h_cls)
        x = self.ln(x)
        x = self.act(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


# 4. SHARED VERDICT HEAD

class SharedVerdictHead(nn.Module):
    """Single shared linear projection layer scoring hypothesis confidence logits at each [MASK]_j.

    Architecture:
        W_V in R^{1 x D}, b_V in R
        score_j = W_V * h_{mask, j} + b_V

    Using a single shared linear layer enforces uniform inductive bias across all hypotheses,
    preventing arbitrary position-based calibration drift.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, h_masks: torch.Tensor) -> torch.Tensor:
        """Score confidence logits for all hypothesis mask representations.

        Args:
            h_masks: Tensor of shape [batch_size, num_hypotheses, hidden_size].

        Returns:
            verdict_logits: Tensor of shape [batch_size, num_hypotheses].
        """
        # Projects each [batch_size, num_hypotheses, hidden_size] -> [batch_size, num_hypotheses, 1]
        out = self.proj(h_masks)
        return out.squeeze(-1)


# 5. UNIFIED LAYA INVESTIGATION MODEL

@dataclass
class InvestigationModelOutput:
    """Output container for LayaInvestigationModel."""
    tool_logits: torch.Tensor                        # [B, num_slots] (after optional Inverse Masking & Budget Pruning)
    verdict_logits: torch.Tensor                     # [B, num_hypotheses]
    raw_tool_logits: torch.Tensor                    # [B, num_slots] (unmasked model prediction)
    h_cls: torch.Tensor                              # [B, hidden_size]
    h_masks: torch.Tensor                            # [B, num_hypotheses, hidden_size]
    cls_indices: torch.Tensor                        # [B]
    mask_indices: torch.Tensor                       # [B, num_hypotheses]
    extended_attention_mask: Optional[torch.Tensor] = None  # [B, 1, S, S] 4D PoE attention bias
    attentions: Optional[Tuple[torch.Tensor, ...]] = None   # Tuple of attention weight tensors if output_attentions=True


class LayaInvestigationModel(nn.Module):
    """Complete Laya Model Refactoring with Tool MLP Head and Shared Verdict Head.

    Production Design Considerations:
      1. PoE Attention Biasing: When poe_attention_biasing=True and hypothesis_mask is provided,
         constructs a 4D additive attention bias tensor directly in the Transformer Self-Attention
         layers. Attention weights between refuted hypotheses and [CLS]/state_t are forced to 0.0,
         so representations physically cannot be corrupted by eliminated branches.
      2. Gradient Starvation Prevention: raw_tool_logits is strictly preserved with full
         autograd gradients for multi-label loss (Asymmetric Focal Loss). Masking & budget
         pruning are only applied during evaluation/inference (not self.training) or when
         apply_inference_masking=True.
      3. Tokenizer Embedding Synchronization: Automatically resizes encoder embedding matrix
         if len(tokenizer) exceeds current input embeddings, preventing GPU index out of range.
    """

    def __init__(
        self,
        encoder: nn.Module,
        tokenizer: Optional[Any] = None,
        num_slots: int = len(CANONICAL_TOOL_SLOTS),
        num_hypotheses: int = len(CANONICAL_HYPOTHESES),
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
        mask_token_id: Optional[int] = None,
        cls_token_id: Optional[int] = None,
        sep_token_id: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = encoder

        # Point 3: Automatic synchronization of tokenizer embeddings
        if tokenizer is not None:
            if hasattr(encoder, "get_input_embeddings") and hasattr(encoder, "resize_token_embeddings"):
                embeddings = encoder.get_input_embeddings()
                if embeddings is not None and embeddings.num_embeddings < len(tokenizer):
                    encoder.resize_token_embeddings(len(tokenizer))
            if mask_token_id is None:
                mask_token_id = getattr(tokenizer, "mask_token_id", None) or tokenizer.convert_tokens_to_ids("[MASK]")
            if cls_token_id is None:
                cls_token_id = getattr(tokenizer, "cls_token_id", None) or tokenizer.convert_tokens_to_ids("[CLS]")
            if sep_token_id is None:
                sep_token_id = getattr(tokenizer, "sep_token_id", None) or tokenizer.convert_tokens_to_ids("[SEP]")

        hidden_size = encoder.config.hidden_size
        self.hidden_size = hidden_size
        self.num_slots = num_slots
        self.num_hypotheses = num_hypotheses
        self.mask_token_id = mask_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

        # Attach heads
        self.tool_head = ToolMLPHead(
            hidden_size=hidden_size,
            num_slots=num_slots,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )
        self.verdict_head = SharedVerdictHead(hidden_size=hidden_size)

    def extract_hidden_vectors(
        self,
        hidden_states: torch.Tensor,
        cls_indices: torch.Tensor,
        mask_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract h_CLS and h_masks vectors from full hidden_states tensor.

        Args:
            hidden_states: [batch_size, seq_len, hidden_size].
            cls_indices: [batch_size].
            mask_indices: [batch_size, num_hypotheses].

        Returns:
            h_cls: [batch_size, hidden_size].
            h_masks: [batch_size, num_hypotheses, hidden_size].
        """
        batch_size = hidden_states.size(0)
        batch_idx = torch.arange(batch_size, device=hidden_states.device)

        # 1. Extract h_CLS: [batch_size, hidden_size]
        h_cls = hidden_states[batch_idx, cls_indices]

        # 2. Extract h_masks: [batch_size, num_hypotheses, hidden_size] via gather
        idx_expanded = mask_indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        h_masks = torch.gather(hidden_states, dim=1, index=idx_expanded)

        return h_cls, h_masks

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cls_indices: Optional[torch.Tensor] = None,
        mask_indices: Optional[torch.Tensor] = None,
        hypothesis_mask: Optional[Union[HypothesisMask, torch.Tensor]] = None,
        budget_k: Optional[int] = None,
        tool_registry: Optional[ToolRegistry] = None,
        cost_aware: bool = False,
        apply_inference_masking: Optional[bool] = None,
        poe_attention_biasing: bool = True,
        output_attentions: Optional[bool] = None,
    ) -> InvestigationModelOutput:
        """Forward pass through encoder with PoE Attention Biasing, Tool MLP Head, and Shared Verdict Head.

        Args:
            input_ids: Tensor of shape [batch_size, seq_len].
            attention_mask: Tensor of shape [batch_size, seq_len].
            cls_indices: Optional explicit [CLS] token indices [batch_size].
            mask_indices: Optional explicit [MASK] token indices [batch_size, num_hypotheses].
            hypothesis_mask: Optional HypothesisMask (M_j in {0, 1}) for PoE Attention Biasing & Inverse Masking.
            budget_k: Optional budget constraint K_t (e.g. K_2=2, K_3=1).
            tool_registry: Optional ToolRegistry for cost-aware ranking.
            cost_aware: If True, ranks candidate slots by utility / cost.
            apply_inference_masking: If True, explicitly applies Inverse Masking & Budget Pruning.
                                     If None (default), applies only during evaluation (not self.training).
                                     If False, strictly bypasses masking so all logits have full gradients.
            poe_attention_biasing: If True and hypothesis_mask is provided, builds a 4D attention bias
                                   tensor blocking attention from refuted hypotheses inside the Transformer.
            output_attentions: Whether to return self-attention weights from encoder.

        Returns:
            InvestigationModelOutput dataclass.
        """
        # 1. Extract token positions if not provided
        if cls_indices is None or mask_indices is None:
            if self.mask_token_id is None:
                raise ValueError("mask_token_id must be configured in model to extract mask positions automatically.")
            cls_indices, mask_indices = PositionTracker.extract_positions(
                input_ids=input_ids,
                mask_token_id=self.mask_token_id,
                cls_token_id=self.cls_token_id,
                expected_num_masks=self.num_hypotheses,
            )

        # 2. Inject PoE Attention Bias into the Encoder Self-Attention mechanism
        extended_attention_mask: Optional[torch.Tensor] = None
        encoder_attention_mask: Optional[torch.Tensor] = attention_mask

        if poe_attention_biasing and hypothesis_mask is not None:
            extended_attention_mask = build_poe_attention_bias(
                input_ids=input_ids,
                mask_indices=mask_indices,
                hypothesis_mask=hypothesis_mask,
                cls_indices=cls_indices,
                base_attention_mask=attention_mask,
                sep_token_id=self.sep_token_id,
                mask_value=SAFE_FP16_MASK_VALUE,
            )
            encoder_attention_mask = extended_attention_mask

        # 3. Encoder forward pass with attention biasing
        encoder_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": encoder_attention_mask,
        }
        if output_attentions is not None:
            encoder_kwargs["output_attentions"] = output_attentions

        encoder_out = self.encoder(**encoder_kwargs)
        hidden_states = encoder_out.last_hidden_state
        attentions = getattr(encoder_out, "attentions", None)

        # 4. Extract h_CLS and h_masks
        h_cls, h_masks = self.extract_hidden_vectors(hidden_states, cls_indices, mask_indices)

        # Gradient Isolation for Refuted Hypotheses:
        # If hypothesis H_j is refuted (M_j = 0), detach h_masks[:, j] so no backward gradient
        # from the verdict head can ever propagate back through the isolated [MASK]_j token
        # into the shared Transformer layers and embeddings.
        if hypothesis_mask is not None:
            if hasattr(hypothesis_mask, "to_list"):
                m_list = hypothesis_mask.to_list()
            elif isinstance(hypothesis_mask, (list, tuple)):
                m_list = list(hypothesis_mask)
            elif isinstance(hypothesis_mask, torch.Tensor):
                m_list = hypothesis_mask.tolist()
            else:
                m_list = [1] * self.num_hypotheses

            if len(m_list) > 0 and isinstance(m_list[0], (int, float)):
                survivor_mask = torch.tensor(m_list, dtype=torch.bool, device=h_masks.device)
                if h_masks.dim() == 3:
                    survivor_mask = survivor_mask.unsqueeze(0).expand(h_masks.size(0), -1)
                h_masks = torch.where(survivor_mask.unsqueeze(-1), h_masks, h_masks.detach())
            elif len(m_list) > 0 and isinstance(m_list[0], list):
                survivor_mask = torch.tensor(m_list, dtype=torch.bool, device=h_masks.device)
                h_masks = torch.where(survivor_mask.unsqueeze(-1), h_masks, h_masks.detach())

        # 5. Predict raw tool logits and verdict logits (always fully differentiable)
        raw_tool_logits = self.tool_head(h_cls)
        verdict_logits = self.verdict_head(h_masks)

        # 6. Determine whether to apply inference masking
        if apply_inference_masking is None:
            # By default: mask only in eval/inference mode; during training keep logits unmasked
            should_mask = (not self.training) and (hypothesis_mask is not None or budget_k is not None)
        else:
            should_mask = apply_inference_masking and (hypothesis_mask is not None or budget_k is not None)

        tool_logits = raw_tool_logits.clone()
        final_verdict_logits = verdict_logits.clone()
        if should_mask:
            batch_size = raw_tool_logits.size(0)

            # Hard Logit Masking for Verdict Head: z_j = -1e4 if M_j == 0
            if hypothesis_mask is not None:
                if hasattr(hypothesis_mask, "to_list"):
                    m_list = hypothesis_mask.to_list()
                elif isinstance(hypothesis_mask, (list, tuple)):
                    m_list = list(hypothesis_mask)
                elif isinstance(hypothesis_mask, torch.Tensor):
                    m_list = hypothesis_mask.tolist()
                else:
                    m_list = [1] * self.num_hypotheses

                if len(m_list) > 0 and isinstance(m_list[0], (int, float)):
                    m_tensor = torch.tensor(m_list, dtype=torch.bool, device=final_verdict_logits.device)
                    if final_verdict_logits.dim() == 2:
                        m_tensor = m_tensor.unsqueeze(0).expand(batch_size, -1)
                    final_verdict_logits = final_verdict_logits.masked_fill(~m_tensor, SAFE_FP16_MASK_VALUE)
                elif len(m_list) > 0 and isinstance(m_list[0], list):
                    m_tensor = torch.tensor(m_list, dtype=torch.bool, device=final_verdict_logits.device)
                    final_verdict_logits = final_verdict_logits.masked_fill(~m_tensor, SAFE_FP16_MASK_VALUE)
            for b in range(batch_size):
                # Map tensor row to slot dict
                row_dict = {
                    CANONICAL_TOOL_SLOTS[i]: float(tool_logits[b, i].item())
                    for i in range(self.num_slots)
                }

                # Inverse Tool Masking (PoE)
                if hypothesis_mask is not None:
                    row_dict = apply_inverse_tool_logits_mask(
                        row_dict,
                        hypothesis_mask=hypothesis_mask,
                        mask_value=SAFE_FP16_MASK_VALUE,
                    )

                # Budget Pruning K_t
                if budget_k is not None:
                    row_dict = prune_slots_by_budget(
                        row_dict,
                        budget_k=budget_k,
                        tool_registry=tool_registry,
                        cost_aware=cost_aware,
                        mask_value=SAFE_FP16_MASK_VALUE,
                    )

                # Write back into tensor
                for i in range(self.num_slots):
                    tool_logits[b, i] = row_dict[CANONICAL_TOOL_SLOTS[i]]

        return InvestigationModelOutput(
            tool_logits=tool_logits,
            verdict_logits=final_verdict_logits,
            raw_tool_logits=raw_tool_logits,
            h_cls=h_cls,
            h_masks=h_masks,
            cls_indices=cls_indices,
            mask_indices=mask_indices,
            extended_attention_mask=extended_attention_mask,
            attentions=attentions,
        )
