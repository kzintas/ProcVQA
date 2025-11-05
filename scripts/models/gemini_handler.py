"""
Gemini 2.5 Pro model handler for image event extraction.
"""

from typing import Tuple, Any, Dict, Optional, Union

from .base_model_handler import BaseModelHandler
from .utils.response_processor import process_llm_response
from .utils.schemas import EventResponse
from google import genai
from google.genai import types
from google.genai.types import Part, GenerateContentConfig, ThinkingConfig
import os
import base64
from pathlib import Path


class GeminiHandler(BaseModelHandler):
    """Handler for Gemini 2.5 Pro model."""

    def __init__(
        self,
        model_id: str = "gemini-2.5-pro-preview-05-06",
        cache_dir: str = None,
        api_key: str = None,
        use_vertex_ai: bool = False,
        project_id: str = None,
        location: str = "us-central1",
        enable_thinking: bool = False,
        thinking_budget: int = 1024,
        **kwargs,
    ):
        """
        Initialize the Gemini model handler.

        Args:
            model_id: The model identifier (default: "gemini-2.5-pro-preview-05-06")
            cache_dir: Directory to cache model files (not used for API-based models)
            api_key: Google API key (if use_vertex_ai is False)
            use_vertex_ai: Whether to use Vertex AI (Google Cloud) instead of API key
            project_id: Google Cloud project ID (required if use_vertex_ai is True)
            location: Google Cloud location (for Vertex AI)
            enable_thinking: Whether to enable thinking mode (default: False)
            thinking_budget: Token budget for thinking (default: 1024, use 0 to turn off)
            **kwargs: Additional arguments for model initialization
        """
        super().__init__(model_id, cache_dir)
        self.api_key = api_key or os.environ.get("$GEMINI_API_KEY")
        self.use_vertex_ai = use_vertex_ai
        self.project_id = project_id or os.environ.get("$VERTEX_PROJECT_ID")
        self.location = location
        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget
        self.kwargs = kwargs
        self.client = None

    def load_model(self) -> Tuple[Any, Any]:
        """
        Initialize the Gemini API client.

        Returns:
            Tuple of (None, None) as this is an API-based model
        """
        if self.use_vertex_ai:
            if not self.project_id:
                raise ValueError("project_id must be provided when use_vertex_ai=True")
            self.client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.location,
            )
        else:
            if not self.api_key:
                raise ValueError("api_key must be provided when use_vertex_ai=False")
            self.client = genai.Client(api_key=self.api_key)

        return None, None

    def _create_generation_config(
        self, sys_prompt: str = None
    ) -> GenerateContentConfig:
        """
        Create a GenerateContentConfig with optional thinking support.

        Args:
            sys_prompt: System instruction to include

        Returns:
            GenerateContentConfig object
        """
        config_kwargs = {}

        if sys_prompt:
            config_kwargs["system_instruction"] = sys_prompt

        if self.enable_thinking and self.thinking_budget > 0:
            config_kwargs["thinking_config"] = ThinkingConfig(
                thinking_budget=self.thinking_budget
            )

        return GenerateContentConfig(**config_kwargs)

    def generate(
        self,
        image_path: str,
        sys_prompt: str = None,
        user_prompt: str = None,
        extraction: bool = True,
    ) -> Tuple[str, Any, Optional[Dict]]:
        """
        Generate event response using the Gemini model.

        Args:
            image_path: Path to the image file
            sys_prompt: System prompt to use
            user_prompt: User prompt to use
            extraction: Whether to extract structured data from the response

        Returns:
            Tuple of (raw_output, processed_output, thinking_info)
            - raw_output: The text response from the model
            - processed_output: Structured data if extraction=True, else raw text
            - thinking_info: Dict with thinking token counts if thinking is enabled, else None
        """
        # Initialize the client if not already done
        if self.client is None:
            self.load_model()

        try:
            # Handle the image - Gemini has different approaches depending on API version
            if self.use_vertex_ai:
                # For Vertex AI-based models
                with open(image_path, "rb") as f:
                    image_data = f.read()

                # Create the request
                response = self.client.models.generate_content(
                    model=self.model_id,
                    config=self._create_generation_config(sys_prompt),
                    contents=[
                        Part.from_bytes(
                            data=image_data, mime_type=self._get_mime_type(image_path)
                        ),
                        user_prompt,
                    ],
                )
            else:
                # For regular API key-based models
                # Upload the file to Gemini
                my_file = self.client.files.upload(file=image_path)

                # Generate content
                response = self.client.models.generate_content(
                    model=self.model_id,
                    config=self._create_generation_config(sys_prompt),
                    contents=[my_file, user_prompt],
                )

            # Extract the text response
            output_text = response.text

            # Extract thinking information if available
            thinking_info = None
            if self.enable_thinking and hasattr(response, "usage_metadata"):
                thinking_info = {
                    "thoughts_token_count": getattr(
                        response.usage_metadata, "thoughts_token_count", 0
                    ),
                    "total_token_count": getattr(
                        response.usage_metadata, "total_token_count", 0
                    ),
                    "thinking_enabled": True,
                }

            # Process the response if extraction is requested
            if extraction:
                processed_response = process_llm_response(output_text)
            else:
                processed_response = output_text

            return output_text, processed_response, thinking_info

        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            return f"Error: {str(e)}", None, None

    def _get_mime_type(self, image_path: str) -> str:
        """Determine the MIME type based on the image file extension"""
        if image_path.lower().endswith(".png"):
            return "image/png"
        elif image_path.lower().endswith((".jpg", ".jpeg")):
            return "image/jpeg"
        else:
            # Default to PNG
            return "image/png"

    def encode(self, path: str | Path) -> str:
        """
        Encode an image file to base64 for API transmission.

        Args:
            path: Path to the image file

        Returns:
            Base64 encoded image with MIME type
        """
        mime_type = self._get_mime_type(str(path))
        return (
            f"data:{mime_type};base64,"
            + base64.b64encode(Path(path).read_bytes()).decode()
        )

    def _get_default_prompt(self) -> str:
        """
        Get the default prompt template for Gemini.

        Returns:
            The prompt template string
        """
        return """No Prompt"""


# Example usage with thinking mode:
"""
# Initialize handler with thinking enabled
handler = GeminiHandler(
    model_id="gemini-2.5-flash",
    api_key="your-api-key",
    enable_thinking=True,
    thinking_budget=1024  # Set to 0 to disable thinking
)

# Generate response
raw_output, processed_output, thinking_info = handler.generate(
    image_path="path/to/image.png",
    sys_prompt="You are a helpful assistant.",
    user_prompt="What's in this image?",
    extraction=False
)

# Check thinking information
if thinking_info:
    print(f"Thoughts token count: {thinking_info['thoughts_token_count']}")
    print(f"Total token count: {thinking_info['total_token_count']}")
"""
