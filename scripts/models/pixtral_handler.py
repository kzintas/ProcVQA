from typing import Tuple, Any, Dict
import torch
import gc

from .base_model_handler import BaseModelHandler
from .utils.response_processor import process_pixtral_response
from .utils.schemas import EventResponse


class PixtralHandler(BaseModelHandler):
    """Handler for Pixtral model."""
    def __init__(
        self, 
        model_id: str = "mistral-community/pixtral-12b",
        cache_dir: str = "/fs/clip-projects/bus_res/HuggingFace/",
        use_bfloat16: bool = True,
        **kwargs
    ):
        """
        Initialize the Pixtral model handler.
        
        Args:
            model_id: The model identifier (default: "mistral-community/pixtral-12b")
            cache_dir: Directory to cache model files
            use_bfloat16: Whether to use bfloat16 precision to reduce memory usage
            **kwargs: Additional arguments for model initialization
        """
        super().__init__(model_id, cache_dir)
        self.use_bfloat16 = use_bfloat16
        self.kwargs = kwargs

        
    def load_model(self):
        """Load Pixtral model and processor."""
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id, 
            device_map="auto", 
            cache_dir=self.cache_dir
        )
        return self.processor, self.model
    
    def generate(self, image_path: str, sys_prompt: str, user_prompt: str , extraction: bool = True):
        """Generate event response using the Pixtral model."""
        chat = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "content": sys_prompt + "\n\n" + user_prompt 
                    },
                    {
                        "type": "image", "url": image_path}
                ]
            }
        ]
        
        # Create input tokens for the model
        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate text
        generate_ids = self.model.generate(**inputs, max_new_tokens=1024)
        output_text = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Process the response
        return output_text, process_pixtral_response(output_text) if extraction else output_text

    def _get_chat_template_pixtral(self):
        return {
              "chat_template": "{%- if messages[0][\"role\"] == \"system\" %}{%- if messages[0][\"content\"] is not string %}{%- if messages[0][\"content\"]|length == 1 and messages[0][\"content\"][0][\"type\"] == \"text\" %}{%- set system_message = messages[0][\"content\"][0][\"content\"] %}{%- endif %}{%- else %}{%- set system_message = messages[0][\"content\"] %}{%- endif %}{%- set loop_messages = messages[1:] %}\n{%- else %}{%- set loop_messages = messages %}{%- endif %}{{- bos_token }}{%- for message in loop_messages %}{%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}{%- endif %}{%- if message[\"role\"] == \"user\" %}{%- if loop.first and system_message is defined %}{{- \"[INST]\" + system_message + \"\n\n\" }}{%- else %}{{ \"[INST]\" }}{%- endif %}{%- endif %}{%- if message[\"content\"] is not string %}{%- for chunk in message[\"content\"] %}{%- if chunk[\"type\"] == \"text\" %}{%- if \"content\" in chunk %}{{- chunk[\"content\"] }}{%- elif \"text\" in chunk %}{{- chunk[\"text\"] }}{%- endif %}{%- elif chunk[\"type\"] == \"image\" %}{{- \"[IMG]\" }}{%- else %}{{- raise_exception(\"Unrecognized content type!\") }}{%- endif %}{%- endfor %}{%- else %}{{- message[\"content\"] }}{%- endif %}{%- if message[\"role\"] == \"user\" %}{{- \"[/INST]\" }}{%- elif message[\"role\"] == \"assistant\" %}{{- eos_token}}{%- else %}{{- raise_exception(\"Only user and assistant roles are supported, with the exception of an initial optional system message!\") }}{%- endif %}{%- endfor %}"
            }
        
    def _get_default_prompt(self) -> str:
        """
        Get the default prompt template for Qwen.
        
        Returns:
            The prompt template string
        """
        return """ None """