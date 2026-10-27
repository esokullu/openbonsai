"""Per-architecture adapters behind one ModelAdapter interface."""
from .base import LayerPlan, ModelAdapter, PlanError
from .qwen3_5 import Qwen35Adapter
from .registry import adapter_for_config, adapter_for_model, load_adapter, register, supported_model_types

__all__ = ["LayerPlan", "ModelAdapter", "PlanError", "Qwen35Adapter", "adapter_for_config",
           "adapter_for_model", "load_adapter", "register", "supported_model_types"]
