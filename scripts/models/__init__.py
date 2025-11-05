"""
Vision-language model handlers for event extraction from images.
"""

from .base_model_handler import BaseModelHandler
from .pixtral_handler import PixtralHandler
from .qwen_handler import QwenHandler
from .gemma3_handler import Gemma3Handler
from .qwenvlmax_handler import QwenvlmaxHandler
from .qwen72B_handler import Qwen72BHandler
from .llamavision11B_handler import Llamavision11BHandler
from .qwen32B_handler import Qwen32BHandler
from .qwenvlplus_handler import QwenvlplusHandler
from .gemini_handler import GeminiHandler

# Define what's available with "from models import *"
__all__ = [
    "BaseModelHandler",
    "PixtralHandler",
    "QwenHandler",
    "Gemma3Handler",
    "QwenvlmaxHandler",
    "QwenvlplusHandler",
    "Qwen72BHandler",
    "Qwen32BHandler",
    "Llamavision11BHandler",
    "GeminiHandler",
    "get_model_handler",
]


# Convenient factory function to get the right handler
def get_model_handler(model_name: str, **kwargs):
    """
    Get the appropriate model handler for the specified model.

    Args:
        model_name: Name of the model ('pixtral', 'qwen', or 'gemma3')
        **kwargs: Additional arguments to pass to the handler

    Returns:
        An instance of the appropriate model handler

    Raises:
        ValueError: If an unknown model name is provided
    """
    handlers = {
        "pixtral": PixtralHandler,
        "qwen": QwenHandler,
        "gemma3": Gemma3Handler,
        "qwenvlmax": QwenvlmaxHandler,
        "qwenvlplus": QwenvlplusHandler,
        "qwen72b": Qwen72BHandler,
        "qwen32b": Qwen32BHandler,
        "llama11b": Llamavision11BHandler,
        "gemini": GeminiHandler,
    }

    if model_name.lower() not in handlers:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(handlers.keys())}"
        )

    return handlers[model_name.lower()](**kwargs)
