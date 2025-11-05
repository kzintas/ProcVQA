"""
Gemma3-27B model handler for image event extraction.
"""

from typing import Tuple, Any, Dict

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import gc
# Import the required transformer modules
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from .base_model_handler import BaseModelHandler
from .utils.response_processor import process_llm_response
from .utils.schemas import EventResponse

class Gemma3Handler(BaseModelHandler):
    """Handler for Gemma3-27B model."""
    
    def __init__(
        self, 
        model_id: str = "google/gemma-3-27b-it",
        cache_dir: str = "/fs/clip-projects/bus_res/HuggingFace/",
        use_bfloat16: bool = True,
        **kwargs
    ):
        """
        Initialize the Gemma3 model handler.
        
        Args:
            model_id: The model identifier (default: "google/gemma-3-27b-it")
            cache_dir: Directory to cache model files
            use_bfloat16: Whether to use bfloat16 precision to reduce memory usage
            **kwargs: Additional arguments for model initialization
        """
        super().__init__(model_id, cache_dir)
        self.use_bfloat16 = use_bfloat16
        self.kwargs = kwargs
    
    def load_model(self) -> Tuple[Any, Any]:
        """
        Load Gemma3 model and processor.
        
        Returns:
            Tuple of (processor, model)
        """
        # Load the model
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_id, 
            device_map="auto", 
            cache_dir=self.cache_dir,
            **self.kwargs
        ).eval()  # Set to evaluation mode
        
        # Load the processor
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        
        return self.processor, self.model
    
    def generate(self, image_path: str, sys_prompt: str = None, user_prompt: str = None, extraction: bool = True):
        """
        Generate event response using the Gemma3 model.
        
        Args:
            image_path: Path to the image file
            prompt: Optional custom prompt to use instead of the default
            
        Returns:
            Dictionary with extracted events
        """
        if self.processor is None or self.model is None:
            raise RuntimeError("Model and processor not loaded. Call load_model() first.")
        
        # Clean up before generation to ensure maximum available memory
        self.cleanup()
        
        # Use custom prompt if provided, otherwise use default
        #prompt_text = prompt if prompt is not None else self._get_default_prompt()
        
        # Format messages for Gemma3
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        # Prepare inputs
        inputs = self.processor.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        )
        
        # Move inputs to device with appropriate dtype
        dtype = torch.bfloat16 if self.use_bfloat16 else torch.float32
        inputs = inputs.to(self.model.device, dtype=dtype)
        
        # Record the input length to extract only new tokens later
        input_len = inputs["input_ids"].shape[-1]
        
        # Generate with torch.inference_mode for memory efficiency
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs, 
                max_new_tokens=1024, 
                do_sample=False
            )
            # Extract only newly generated tokens
            generation = generation[0][input_len:]
        
        # Decode the generated tokens
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        
        # Process the response to extract events
        if extraction: 
            response = process_llm_response(decoded)
        else:
            response = decoded
        
        # Clean up resources
        del inputs, generation
        self.cleanup()
        
        return decoded, response
    
    def _get_default_prompt(self) -> str:
        """
        Get the default prompt template for Gemma3.
        
        Returns:
            The prompt template string
        """
        return """Please extract *all* the event and number pairs from this image.

INSTRUCTIONS:
1. Identify each key text label and its associated number value
2. Return ONLY a valid JSON object with text as keys and numbers as values
3. Do not include any explanations, notes, or additional text
4. Ensure all values are integers, not strings
5. Extract ALL values that you can extract from the image
6. Format your entire response as a single JSON object

EXAMPLE FORMAT:
{<Event_name>: <Event_value>, ...}

IMPORTANT: Your entire response must be a parseable JSON object and nothing else."""