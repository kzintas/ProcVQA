"""
Llama model handler for the image event extraction.
"""

from typing import Tuple, Any, Dict, Optional, Union
import os
import base64
from pathlib import Path

from .base_model_handler import BaseModelHandler
from .utils.response_processor import process_llm_response
from .utils.schemas import EventResponse

import openai
from google.auth import default, transport


class LlamaHandler(BaseModelHandler):
    """Handler for Llama vision models."""

    def __init__(
        self,
        model_id: str = "meta/llama-3.2-90b-vision-instruct-maas",
        cache_dir: str = None,
        project_id: str = None,
        location: str = "us-central1",
        llama_version: str = "3.2",
        max_tokens: int = 4096,
        temperature: float = 0.0,
        **kwargs,
    ):
        """
        Initialize the Llama model handler.

        Args:
            model_id: The model identifier (default: "meta/llama-3.2-90b-vision-instruct-maas")
            cache_dir: Directory to cache model files (not used for API-based models)
            project_id: Google Cloud project ID (required)
            location: Google Cloud location for the model (default: "us-central1")
            llama_version: Version of Llama (determines endpoint, defaults to "3.2")
            max_tokens: Maximum number of tokens for generation
            temperature: Temperature for generation (higher = more random)
            **kwargs: Additional arguments for model initialization
        """
        super().__init__(model_id, cache_dir)
        self.project_id = project_id or os.environ.get("GCP_PROJECT_ID")
        self.location = location
        self.llama_version = llama_version
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.client = None
        self.credentials = None
        self.kwargs = kwargs

        # Configure endpoints based on version
        if not self.location:
            if self.llama_version == "4":
                self.model_location = "us-east5"  # Llama 4 is in East5
            else:
                self.model_location = "us-central1"  # Llama 3 is in Central1

        self.maas_endpoint = f"{self.model_location}-aiplatform.googleapis.com"

    def load_model(self) -> Tuple[Any, Any]:
        """
        Initialize the Llama client and authentication.

        Returns:
            Tuple of (None, None) as this is an API-based model
        """
        if not self.project_id:
            raise ValueError("project_id must be provided for Llama models")

        # Get GCP credentials
        self.credentials, _ = default()
        auth_request = transport.requests.Request()
        self.credentials.refresh(auth_request)

        # Initialize OpenAI client with Vertex AI endpoint
        self.client = openai.OpenAI(
            base_url=f"https://{self.maas_endpoint}/v1beta1/projects/{self.project_id}/locations/{self.model_location}/endpoints/openapi",
            api_key=self.credentials.token,
        )

        return None, None

    def generate(
        self,
        image_path: str,
        sys_prompt: str = None,
        user_prompt: str = None,
        extraction: bool = True,
    ) -> Dict:
        """
        Generate event response using the Llama model.

        Args:
            image_path: Path to the image file
            sys_prompt: System prompt to use
            user_prompt: User prompt to use
            extraction: Whether to extract structured data from the response

        Returns:
            Tuple of (raw_output, processed_output)
        """
        # Initialize the client if not already done
        if self.client is None:
            self.load_model()

        try:
            # Open and read the image file as binary data
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()

            # Convert binary data to base64 encoding
            base64_image = base64.b64encode(image_data).decode("utf-8")

            # Create the message with system prompt and base64-encoded image
            messages = [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]

            # Make the API call
            raw_response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                **self.kwargs,
            )

            # Process the response
            output_text = raw_response.choices[0].message.content

            # Process the response if requested
            if extraction:
                processed_response = process_llm_response(output_text)
            else:
                processed_response = output_text

            return output_text, processed_response

        except Exception as e:
            print(f"Error generating content with Llama: {e}")
            return f"Error: {str(e)}", None

    def encode(self, path: str | Path) -> str:
        """
        Encode an image file to base64 for API transmission.

        Args:
            path: Path to the image file

        Returns:
            Base64 encoded image with MIME type
        """
        mime = "image/png" if str(path).lower().endswith(".png") else "image/jpeg"
        return (
            f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()
        )

    def _get_default_prompt(self) -> str:
        """
        Get the default prompt template for Llama.

        Returns:
            The prompt template string
        """
        return """No Prompt"""

    def refresh_credentials(self):
        """
        Refresh Google Cloud credentials, useful for long-running processes.
        """
        if self.credentials:
            auth_request = transport.requests.Request()
            self.credentials.refresh(auth_request)

            # Update client with new token
            self.client = openai.OpenAI(
                base_url=f"https://{self.maas_endpoint}/v1beta1/projects/{self.project_id}/locations/{self.model_location}/endpoints/openapi",
                api_key=self.credentials.token,
            )
