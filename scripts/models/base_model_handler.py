"""
Base model handler for vision-language models.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Any, Dict
import torch
import gc


class BaseModelHandler(ABC):
    """
    Abstract base class for all vision-language model handlers.
    
    This class defines the interface that all model handlers must implement.
    """
    
    def __init__(self, model_id: str, cache_dir: str):
        """
        Initialize the base model handler.
        
        Args:
            model_id: The model identifier
            cache_dir: Directory to cache model files
        """
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.processor = None
        self.model = None
    
    @abstractmethod
    def load_model(self) -> Tuple[Any, Any]:
        """
        Load the model and processor.
        
        Returns:
            Tuple of (processor, model)
        """
        pass
        
    @abstractmethod
    def generate(self, image_path: str, prompt: str = None) -> Dict:
        """
        Generate output from an image with an optional prompt.
        
        Args:
            image_path: Path to the image file
            prompt: Optional prompt to guide the generation
            
        Returns:
            Dictionary with the generated output
        """
        pass
    
    @abstractmethod
    def _get_default_prompt(self) -> str:
        """
        Get the default prompt template for this model.
        
        Returns:
            The default prompt template string
        """
        pass
    
    def cleanup(self) -> None:
        """
        Clean up resources after generation to prevent memory leaks.
        
        This is especially important when working with large models
        and processing multiple images in sequence.
        """
        torch.cuda.empty_cache()
        gc.collect()