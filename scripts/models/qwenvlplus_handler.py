"""
Qwen2.5-VL model handler for the image event extraction.
"""
from typing import Tuple, Any, Dict

from .base_model_handler import BaseModelHandler
from .utils.response_processor import process_llm_response
from .utils.schemas import EventResponse
from openai import OpenAI
import os
import base64, json, os, openai
from pathlib import Path

class QwenvlplusHandler(BaseModelHandler):
    """Handler for Qwen2.5-VL model."""
    
    def __init__(
        self, 
        model_id: str = "qwen-vl-plus",
        cache_dir: str = "/fs/clip-projects/bus_res/HuggingFace/",
        load_in_8bit: bool = True,
        **kwargs
    ):
        """
        Initialize the Qwen model handler.
        
        Args:
            model_id: The model identifier (default: "Qwen/Qwen2.5-VL-7B-Instruct")
            cache_dir: Directory to cache model files
            load_in_8bit: Whether to use 8-bit quantization to reduce memory usage
            **kwargs: Additional arguments for model initialization
        """
        super().__init__(model_id, cache_dir)
        self.load_in_8bit = load_in_8bit
        self.kwargs = kwargs
    
    def load_model(self) -> Tuple[Any, Any]:
        """
        Load Qwen2.5-VL model and processor.
        
        Returns:
            Tuple of (processor, model)
        """
        pass
        # try:
        #     from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            
        #     # Load the model with appropriate memory optimizations
        #     self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #         self.model_id,
        #         device_map="balanced",  # Balance between GPU and CPU
        #         offload_folder="offload",  # Enable CPU offloading
        #         load_in_8bit=self.load_in_8bit,  # Use 8-bit quantization
        #         cache_dir=self.cache_dir,
        #         **self.kwargs
        #     )
            
        #     # Load the processor
        #     self.processor = AutoProcessor.from_pretrained(self.model_id)
            
        #     return self.processor, self.model
            
        # except ImportError:
        #     raise ImportError(
        #         "Could not import required modules. "
        #         "Make sure transformers is installed with: pip install transformers"
        #     )
    
    def generate(self, image_path: str, sys_prompt: str = None, user_prompt: str = None, extraction: bool = True) -> Dict:
        """
        Generate event response using the Qwen2.5-VL model.
        
        Args:
            image_path: Path to the image file
            prompt: Optional custom prompt to use instead of the default
            
        Returns:
            Dictionary with extracted events
        """
        client = OpenAI(
            api_key="", ## ENTER YOUR DASHSCOPE API KEY HERE
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        
        completion = client.chat.completions.create(
            model="qwen-vl-plus",  # Using qwen-vl-max as an example, you can change the model name as needed. Model list:  https://www.alibabacloud.com/help/model-studio/getting-started/models
            messages=[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": sys_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self.encode(image_path)
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
        )

        output_text = completion.choices[0].message.content

        # Process the response
        if extraction: 
            response = process_llm_response(output_text)
        else:
            response = output_text
            
        
        return output_text, response

    def encode(self, path: str | Path) -> str:
        mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()

    
    def _get_default_prompt(self) -> str:
        """
        Get the default prompt template for Qwen.
        
        Returns:
            The prompt template string
        """
        return """ No Prompt """