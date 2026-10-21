"""Incremental evidence memory manager and KV-cache extension for Rex (Step A4)."""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .rex_ponder import LatentPonderBlock
from .rex_types import RexState


class EvidenceMemoryManager:
    """Manages multi-round evidence expansion and incremental cross-attention KV caching."""

    @staticmethod
    def append_evidence(
        state: RexState,
        M_new: torch.Tensor,
        memory_mask_new: torch.Tensor,
        is_truncated: bool = False,
        ponder_block: Optional[LatentPonderBlock] = None,
    ) -> RexState:
        """Append newly contextualized evidence piece M_new to running memory M.

        Properties:
            - h is strictly preserved across tool turns (never reset to h_0).
            - M is extended along sequence dimension: M_{k+1} = [M_k ; M_new].
            - memory_mask is concatenated accordingly.
            - If ponder_block and kv_cache are available, cache is updated incrementally in O(L_new).

        Args:
            state: Current RexState.
            M_new: Contextualized token representations of new observation [B, L_new, d].
            memory_mask_new: Validity mask [B, L_new] bool.
            is_truncated: Whether observation was capped at 64 tokens.
            ponder_block: Optional LatentPonderBlock instance to extend KV cache.

        Returns:
            Updated RexState instance with expanded memory and updated cache.
        """
        B, L_new, d = M_new.shape

        if state.M.size(1) == 0:
            # First assignment (or transition from empty state)
            new_M = M_new
            new_mask = memory_mask_new
            new_kv = ponder_block.build_kv_cache(M_new) if ponder_block is not None else None
        else:
            new_M = torch.cat([state.M, M_new], dim=1)
            new_mask = torch.cat([state.memory_mask, memory_mask_new], dim=1)

            # Extend KV-cache incrementally if cache already exists
            if state.kv_cache is not None and ponder_block is not None:
                new_kv = ponder_block.extend_kv_cache(state.kv_cache, M_new)
            elif ponder_block is not None:
                new_kv = ponder_block.build_kv_cache(new_M)
            else:
                new_kv = state.kv_cache

        state.M = new_M
        state.memory_mask = new_mask
        state.kv_cache = new_kv
        state.is_truncated = state.is_truncated or is_truncated
        return state
