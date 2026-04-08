"""模型注册表 - 统一管理所有模型的实例化"""

from mytools.registry import register_model, build_model, get_model, list_models

_MODEL_REGISTRY = {}

__all__ = ["register_model", "build_model", "get_model", "list_models", "_MODEL_REGISTRY"]
