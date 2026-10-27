"""What a model architecture must provide to the framework.

The core only sees relative keys (``layers.{i}.{sub}``, ``lm_head``,
``embed_tokens``) and the per-layer plan. Rotation geometry (sign widths,
Hadamard block) is derived by the core from the planned linears
(openbonsai.core.geometry), never declared by the adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import nn


@dataclass(frozen=True)
class LayerPlan:
    kind: str                                  # e.g. "linear_attention", "full_attention", "dense"
    groups: tuple[tuple[str, ...], ...]        # sequential GPTQ order; one group shares one input
    keep: tuple[str, ...] = ()                 # Linear submodules kept high precision

    @property
    def quantized(self) -> tuple[str, ...]:
        return tuple(n for g in self.groups for n in g)


class PlanError(ValueError):
    """A layer's Linear modules are not covered exactly once by its plan."""


class ModelAdapter:
    name: ClassVar[str] = "base"
    model_types: ClassVar[tuple[str, ...]] = ()
    gguf_arch: ClassVar[str | None] = None     # Prism GGUF architecture; None = not exportable
    add_bos: ClassVar[bool] = False            # start every packed block / eval sequence with BOS
    zero_centered_norm: ClassVar[bool] = False  # norm is (1 + w) * x_hat instead of w * x_hat

    def __init__(self, config):
        self.config = config
        self.text_config = getattr(config, "text_config", None) or config

    # -- loading ---------------------------------------------------------
    def load(self, path, *, revision=None, device="cpu", dtype=torch.bfloat16, attn="sdpa",
             local_files_only=False, **options):
        """Return (model, tokenizer); the model is in eval mode with grads off."""
        raise NotImplementedError

    def load_tokenizer(self, path, *, revision=None, local_files_only=False):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(path, revision=revision, local_files_only=local_files_only,
                                             trust_remote_code=False)

    # -- structure -------------------------------------------------------
    def text_model(self, model) -> nn.Module:
        """Module with .embed_tokens, .layers and .norm whose forward returns last_hidden_state."""
        raise NotImplementedError

    def head(self, model) -> nn.Module:
        return model.lm_head

    def set_head(self, model, module) -> None:
        model.lm_head = module

    def embedding(self, model) -> nn.Module:
        return self.text_model(model).embed_tokens

    def set_embedding(self, model, module) -> None:
        self.text_model(model).embed_tokens = module

    def layers(self, model) -> nn.ModuleList:
        return self.text_model(model).layers

    @property
    def n_layers(self) -> int:
        return int(self.text_config.num_hidden_layers)

    @property
    def hidden_size(self) -> int:
        return int(self.text_config.hidden_size)

    @property
    def vocab_size(self) -> int:
        return int(self.text_config.vocab_size)

    @property
    def tied(self) -> bool:
        return bool(getattr(self.config, "tie_word_embeddings", False)
                    or getattr(self.text_config, "tie_word_embeddings", False))

    def layer_plan(self, i: int) -> LayerPlan:
        raise NotImplementedError

    # -- numerics --------------------------------------------------------
    def embedding_gain(self, model) -> torch.Tensor:
        """FP32 gain of the first layer's entry norm: what an embedding row is
        multiplied by before layer 0 sees it. GPTQ for the embedding uses
        H = diag(gain^2)."""
        w = self.layers(model)[0].input_layernorm.weight.float()
        return 1.0 + w if self.zero_centered_norm else w

    # -- chat and generation ---------------------------------------------
    def chat_kwargs(self, *, thinking: bool) -> dict:
        return {}

    def eos_ids(self, tokenizer, model) -> list[int]:
        """Model generation-config EOS ids plus the tokenizer's EOS."""
        configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        values = list(configured) if isinstance(configured, (list, tuple)) else (
            [] if configured is None else [configured])
        tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
        if tokenizer_eos is not None:
            values.append(tokenizer_eos)
        return [int(v) for v in dict.fromkeys(values)]

    def bos_id(self, tokenizer) -> int | None:
        if not self.add_bos:
            return None
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is None:
            raise ValueError(f"{self.name} adds BOS but the tokenizer defines none")
        return int(bos)

    # -- plan helpers (enforced by the core) -----------------------------
    def planned_linears(self, model, i: int) -> dict[str, nn.Linear]:
        """Validate layer i's plan against its actual Linear modules."""
        plan = self.layer_plan(i)
        layer = self.layers(model)[i]
        linears = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}
        planned = list(plan.quantized) + list(plan.keep)
        dup = {n for n in planned if planned.count(n) > 1}
        missing = [n for n in planned if n not in linears]
        uncovered = [n for n in linears if n not in planned]
        if dup or missing or uncovered:
            raise PlanError(f"layer {i} ({plan.kind}): duplicated {sorted(dup)}, missing {missing}, "
                            f"unplanned linears {uncovered}")
        return {n: linears[n] for n in plan.quantized}

    def quantized_widths(self, model) -> set[int]:
        widths = set()
        for i in range(self.n_layers):
            widths.update(m.in_features for m in self.planned_linears(model, i).values())
        return widths
