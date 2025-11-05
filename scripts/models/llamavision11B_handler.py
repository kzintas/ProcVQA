"""
Qwen2.5-VL model handler for the image event extraction.
"""
from typing import Tuple, Any, Dict

from .base_model_handler import BaseModelHandler
from .utils.response_processor import process_llm_response
from .utils.schemas import EventResponse
import os
from pathlib import Path
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor


class Llamavision11BHandler(BaseModelHandler):
    """Handler for Llama-3.2-11B-Vision-Instruct model."""
    
    def __init__(
        self, 
        model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        cache_dir: str = "/fs/clip-projects/bus_res/HuggingFace/",
        load_in_8bit: bool = True,
        **kwargs
    ):
        """
        Initialize the Llama model handler.
        
        Args:
            model_id: The model identifier (default: "meta-llama/Llama-3.2-11B-Vision-Instruct")
            cache_dir: Directory to cache model files
            load_in_8bit: Whether to use 8-bit quantization to reduce memory usage
            **kwargs: Additional arguments for model initialization
        """
        super().__init__(model_id, cache_dir)
        self.load_in_8bit = load_in_8bit
        self.kwargs = kwargs
    
    def load_model(self) -> Tuple[Any, Any]:
        """
        Load Llama-Vision-12B model and processor.
        
        Returns:
            Tuple of (processor, model)
        """
        try:
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                cache_dir=self.cache_dir,
                **self.kwargs
            )
            self.processor = AutoProcessor.from_pretrained(self.model_id)

            
            return self.processor, self.model
            
        except ImportError:
            raise ImportError(
                "Could not import required modules. "
                "Make sure transformers is installed with: pip install transformers"
            )
    
    def generate(self, image_path: str, sys_prompt: str = None, user_prompt: str = None, extraction: bool = True) -> Dict:
        """
        Generate event response using the Qwen-VL-Max model.
        
        Args:
            image_path: Path to the image file
            prompt: Optional custom prompt to use instead of the default
            
        Returns:
            Dictionary with extracted events
        """
        image = Image.open(image_path)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": sys_prompt}]
            },
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": user_prompt}
            ]}
        ]

        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        inputs = self.processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(self.model.device)

        output = self.model.generate(**inputs, max_new_tokens=1000, temperature = 0.01)
        prompt_len = inputs.input_ids.shape[-1]
        generated_ids = output[:, prompt_len:]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        #print(output_text)
        
        # Process the response
        if extraction: 
            response = process_llm_response(output_text)
        else:
            response = output_text
            
        
        return output_text, response
    
    def _get_default_prompt(self) -> str:
        """
        Get the default prompt template for Qwen.
        
        Returns:
            The prompt template string
        """
        return """ No Prompt """