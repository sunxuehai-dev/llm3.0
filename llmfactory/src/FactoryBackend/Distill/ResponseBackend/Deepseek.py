from typing import Optional, Tuple
from ..Response import Response


class Deepseek(Response):
    """
    Deepseek model response backend implementation with thinking mode support
    
    Provides a specialized interface for Deepseek model with thinking mode capability.
    """
    
    def __init__(self, *, base_url: str, api_key: str, model: str, enable_thinking: bool = False, **kwargs) -> None:
        """
        Initialize the Deepseek response client
        
        Args:
            base_url: API base URL
            api_key: API key
            model: Model name
            enable_thinking: Whether to enable thinking mode by default
            **kwargs: Additional OpenAI client parameters
        """
        super().__init__(base_url=base_url, api_key=api_key, model=model, **kwargs)
        self._enable_thinking = enable_thinking
    
    def set_enable_thinking(self, enable: bool) -> None:
        """
        Set whether to enable thinking mode
        
        Args:
            enable: True to enable thinking mode, False to disable
        """
        self._enable_thinking = enable
        

    def reply(self, message: str) -> str:
        """
        Synchronously call the API and return the response content
        
        This method always returns the content only, regardless of thinking mode.
        Use reply_with_thinking() to get both reasoning content and content.
        
        Args:
            message: User message
            
        Returns:
            Model response content (without reasoning content)
        """
        if self._enable_thinking:
            _, content = self.reply_with_thinking(message)
            return content
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": message}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
    
    def reply_with_thinking(self, message: str) -> Tuple[str, str]:
        """
        Generate a response and return reasoning content and content separately
        
        Args:
            message: User message
            
        Returns:
            Tuple of (reasoning_content, content)
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": message}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={"thinking": {"type": "enabled"}}
        )
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        return reasoning_content, content
