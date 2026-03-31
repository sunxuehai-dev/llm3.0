"""
Qwen3 model response backend implementation

Provides a specialized interface for Qwen3 model with thinking mode support.
"""

from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..Response import AbstractResponse


class ResponseQwen3(AbstractResponse):
    """
    Qwen3 model response implementation with thinking mode support
    
    Uses transformers library to load and interact with Qwen3 model locally.
    Supports both thinking and non-thinking modes for different use cases.
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str = "Qwen/Qwen3-1.7B",
        enable_thinking: bool = True,
        max_new_tokens: int = 32768,
        torch_dtype: str = "auto",
        device_map: str = "auto",
        **kwargs
    ) -> None:
        """
        Initialize the Qwen3 response client
        
        Args:
            base_url: API base URL (not used for local model, kept for compatibility)
            api_key: API key (not used for local model, kept for compatibility)
            model: HuggingFace model name or local path to the model
                   - For HuggingFace: "Qwen/Qwen3-1.7B"
                   - For local: "/path/to/model" or "./models/qwen3-1.7B"
            enable_thinking: Whether to enable thinking mode by default
            max_new_tokens: Maximum number of tokens to generate
            torch_dtype: Data type for model weights (e.g., "auto", "float16", "bfloat16")
            device_map: Device mapping strategy (e.g., "auto", "cuda", "cpu")
            **kwargs: Additional parameters for model loading
        """
        super().__init__(base_url, api_key, model)
        
        self.enable_thinking = enable_thinking
        self.max_new_tokens = max_new_tokens
        
        # Resolve model path (handle both local paths and HuggingFace model names)
        model_path = self._resolve_model_path(model)
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=True,
            **kwargs
        )
    
    def _resolve_model_path(self, model: str) -> str:
        """
        Resolve model path to either local path or HuggingFace model name
        
        Args:
            model: Model identifier (local path or HuggingFace name)
            
        Returns:
            Resolved model path
            
        Raises:
            FileNotFoundError: If local model path does not exist
        """
        # Check if it's a local path
        if "/" in model or "\\" in model:
            # It might be a local path
            path = Path(model)
            if path.exists():
                return str(path.resolve())
            
            # Try relative to current working directory
            cwd_path = Path.cwd() / model
            if cwd_path.exists():
                return str(cwd_path.resolve())
        
        # Otherwise, treat it as a HuggingFace model name
        return model
    
    def set_enable_thinking(self, enable: bool) -> None:
        """
        Set whether to enable thinking mode
        
        Args:
            enable: True to enable thinking mode, False to disable
        """
        self.enable_thinking = enable
    
    def reply(self, message: str) -> str:
        """
        Generate a response for the given message
        
        Args:
            message: User message
            
        Returns:
            Model response text (content only, thinking content is parsed separately)
        """
        thinking_content, content = self.reply_with_thinking(message)
        
        # Return content only (thinking content can be accessed via reply_with_thinking)
        return content
    
    def reply_with_thinking(self, message: str) -> Tuple[str, str]:
        """
        Generate a response and return thinking content and content separately
        
        Args:
            message: User message
            
        Returns:
            Tuple of (thinking_content, content)
        """
        # Prepare the model input
        messages = [
            {"role": "user", "content": message}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True if self.temperature > 0 else False
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
        # Parse thinking content
        thinking_content, content = self._parse_thinking_content(output_ids)
        
        return thinking_content, content
    
    def _parse_thinking_content(self, output_ids: list) -> Tuple[str, str]:
        """
        Parse thinking content from output token IDs
        
        Args:
            output_ids: List of output token IDs
            
        Returns:
            Tuple of (thinking_content, content)
        """
        # Token ID for  (151668) - this is the separator token
        thinking_token_id = 151668
        
        try:
            # Find the last occurrence of thinking token
            index = len(output_ids) - output_ids[::-1].index(thinking_token_id)
        except ValueError:
            # No thinking token found, all content is regular content
            index = 0
        
        # Decode thinking content and content separately
        thinking_content = self.tokenizer.decode(
            output_ids[:index],
            skip_special_tokens=True
        ).strip("\n")
        
        content = self.tokenizer.decode(
            output_ids[index:],
            skip_special_tokens=True
        ).strip("\n")
        
        return thinking_content, content
    
    def reply_content_only(self, message: str) -> str:
        """
        Generate a response and return only the content (without thinking)
        
        Args:
            message: User message
            
        Returns:
            Model response content without thinking
        """
        _, content = self.reply_with_thinking(message)
        return content
    
    def reply_thinking_only(self, message: str) -> str:
        """
        Generate a response and return only the thinking content
        
        Args:
            message: User message
            
        Returns:
            Model thinking content
        """
        thinking_content, _ = self.reply_with_thinking(message)
        return thinking_content
