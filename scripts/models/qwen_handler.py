"""
Qwen2.5-VL model handler for the image event extraction.
"""
from typing import Tuple, Any, Dict

import torch
import gc
from .base_model_handler import BaseModelHandler
from .utils.response_processor import process_llm_response
from .utils.schemas import EventResponse


class QwenHandler(BaseModelHandler):
    """Handler for Qwen2.5-VL model."""
    
    def __init__(
        self, 
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
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
        try:
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            
            # Load the model with appropriate memory optimizations
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                device_map="balanced",  # Balance between GPU and CPU
                offload_folder="offload",  # Enable CPU offloading
                load_in_8bit=self.load_in_8bit,  # Use 8-bit quantization
                cache_dir=self.cache_dir,
                **self.kwargs
            )
            
            # Load the processor
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            
            return self.processor, self.model
            
        except ImportError:
            raise ImportError(
                "Could not import required modules. "
                "Make sure transformers is installed with: pip install transformers"
            )
    
    def generate(self, image_path: str, sys_prompt: str = None, user_prompt: str = None, extraction: bool = True) -> Dict:
        """
        Generate event response using the Qwen2.5-VL model.
        
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
        sys_prompt_text = sys_prompt if sys_prompt is not None else self._get_default_prompt()
        user_prompt_text = user_prompt if user_prompt is not None else self._get_default_prompt()
        

        # Format the conversation for Qwen
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type":"text",
                        "text": str(sys_prompt)
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": image_path
                    },
                    {
                        "type": "text",
                        "text": user_prompt
                    }
                ]
            }
        ]
        
        # Prepare inputs
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate output
        output_ids = self.model.generate(**inputs, max_new_tokens=1024)
        
        # Extract only the newly generated tokens, not the entire conversation
        generated_ids = [
            out_ids[len(inp_ids):] for inp_ids, out_ids in zip(inputs.input_ids, output_ids)
        ]
        
        # Decode the generated tokens
        output_text_list = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        output_text = output_text_list[0] if output_text_list else ""
        #print(f"Output Text: \n {output_text}")
        
        # Process the response
        if extraction: 
            response = process_llm_response(output_text)
        else:
            response = output_text
            
        #print(f"Parsed Output: \n {response}")
        
        # Clean up to prevent memory leaks
        del inputs, output_ids, generated_ids
        self.cleanup()
        
        return output_text, response
    
    def _get_default_prompt(self) -> str:
        """
        Get the default prompt template for Qwen.
        
        Returns:
            The prompt template string
        """
        return """Please extract all the event and number pairs from this image.
INSTRUCTIONS:
1. Identify each key text label and its associated number value
2. Return ONLY a valid JSON object with text as keys and numbers as values
3. Do not include any explanations, notes, or additional text
4. Ensure all numbers are integers, not strings
5. Format your entire response as a single JSON object
EXAMPLE FORMAT:
{<Event_name>: <Event_value>, ...}
IMPORTANT: Your entire response must be a parseable JSON object and nothing else."""