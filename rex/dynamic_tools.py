"""Dynamic Tool Representation Encoder for MCP and SDK-registered tools.

Computes semantic embeddings in R^1024 for dynamically declared McpToolSchema objects
using ModernBERT encoder, caching projections, and injecting into RexDecisionModel.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from rex.rex_registry import ToolSpec

DEFAULT_TOOL_COST: float = 0.02
DEFAULT_TOOL_LEVEL: str = "READ"


class DynamicToolEncoder:
    """Encodes dynamic MCP tool descriptions into normalized 1024-dim representations."""

    def __init__(self, cache_size: int = 512) -> None:
        self._cache: Dict[str, torch.Tensor] = {}
        self._cache_size = cache_size

    def _hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def encode_descriptions(
        self,
        descriptions: Sequence[str],
        encoder: Optional[Any] = None,
        tok: Optional[Any] = None,
        device: Optional[torch.device] = None,
        d_model: int = 1024,
    ) -> torch.Tensor:
        """Encode a sequence of tool descriptive texts into a normalized [N, d_model] tensor.

        Falls back to deterministic projection if encoder/tok are None (e.g. lightweight tests).
        """
        if not descriptions:
            return torch.empty((0, d_model), device=device)

        device = device or torch.device("cpu")
        embs: List[torch.Tensor] = []
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        for i, text in enumerate(descriptions):
            h = self._hash_text(text)
            if h in self._cache:
                embs.append(self._cache[h].to(device))
            else:
                embs.append(torch.empty(0))  # Placeholder
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            if encoder is not None and tok is not None and hasattr(tok, "encode"):
                try:
                    # Batch encode through tokenizer
                    encoded = tok(
                        uncached_texts,
                        padding=True,
                        truncation=True,
                        max_length=64,
                        return_tensors="pt",
                    )
                    input_ids = encoded["input_ids"].to(device)
                    attn_mask = encoded["attention_mask"].to(device)

                    with torch.no_grad():
                        out = encoder(input_ids=input_ids, attention_mask=attn_mask)
                        hidden = out[0] if isinstance(out, (tuple, list)) else getattr(out, "last_hidden_state", out)
                        # Mean pooling over valid attention tokens
                        mask_exp = attn_mask.unsqueeze(-1).float()
                        sum_emb = (hidden * mask_exp).sum(dim=1)
                        counts = mask_exp.sum(dim=1).clamp(min=1.0)
                        computed = F.normalize(sum_emb / counts, p=2, dim=-1)
                except Exception:
                    # Fallback on tokenizer/encoder failure
                    computed = self._deterministic_fallback(uncached_texts, d_model, device)
            else:
                computed = self._deterministic_fallback(uncached_texts, d_model, device)

            for idx, text, tensor in zip(uncached_indices, uncached_texts, computed):
                tensor_norm = F.normalize(tensor, p=2, dim=-1)
                h = self._hash_text(text)
                if len(self._cache) < self._cache_size:
                    self._cache[h] = tensor_norm.detach().cpu()
                embs[idx] = tensor_norm.to(device)

        return torch.stack(embs, dim=0)

    def _deterministic_fallback(
        self,
        texts: Sequence[str],
        d_model: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Deterministic pseudo-random embeddings derived from SHA256 seed."""
        tensors: List[torch.Tensor] = []
        for t in texts:
            seed_int = int(hashlib.md5(t.encode("utf-8")).hexdigest()[:8], 16)
            g = torch.Generator().manual_seed(seed_int)
            vec = torch.randn(d_model, generator=g, device=device)
            tensors.append(F.normalize(vec, p=2, dim=-1))
        return torch.stack(tensors, dim=0)

    def register_dynamic_tools(
        self,
        model: Any,
        tools_schema: Sequence[Dict[str, Any]],
        tok: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> List[ToolSpec]:
        """Update RexDecisionModel with dynamic tools registered from SDK.

        Args:
            model: RexDecisionModel instance.
            tools_schema: List of dicts matching McpToolSchema from SDK.
            tok: Tokenizer instance.
            device: Target torch device.

        Returns:
            List of constructed ToolSpec objects.
        """
        if not tools_schema:
            return []

        dev = device or getattr(model, "tool_embeddings", torch.empty(0)).device or torch.device("cpu")
        tool_names: List[str] = []
        descriptions: List[str] = []
        tool_specs: List[ToolSpec] = []

        for schema in tools_schema:
            name = str(schema.get("name", "")).strip()
            if not name:
                continue
            desc = str(schema.get("description", "")).strip() or name
            slot = str(schema.get("slot", "SLOT_NETWORK"))
            cost = float(schema.get("cost", DEFAULT_TOOL_COST))
            level = str(schema.get("level", DEFAULT_TOOL_LEVEL))

            tool_names.append(name)
            # Combine slot + description for semantic differentiation
            text_rep = f"[{slot}] {name}: {desc}"
            descriptions.append(text_rep)

            tool_specs.append(
                ToolSpec(
                    name=name,
                    description=desc,
                    level=level,
                    cost=cost,
                    returns=list(schema.get("inputSchema", {}).get("properties", {}).keys()),
                )
            )

        if not tool_names:
            return []

        encoder = getattr(model, "encoder", None)
        d_model = getattr(encoder.config, "hidden_size", 1024) if encoder and hasattr(encoder, "config") else 1024

        embeddings = self.encode_descriptions(
            descriptions=descriptions,
            encoder=encoder,
            tok=tok,
            device=dev,
            d_model=d_model,
        )

        # Update model state
        model.tools = tool_names
        model.tool_to_idx = {name: i for i, name in enumerate(tool_names)}
        model.tool_embeddings = embeddings
        if hasattr(model, "tool_matcher") and hasattr(model.tool_matcher, "set_tool_embeddings"):
            model.tool_matcher.set_tool_embeddings(embeddings)

        return tool_specs
