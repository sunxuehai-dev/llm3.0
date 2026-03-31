"""
Model distillation response interface module

Provides a unified abstract interface supporting both synchronous and asynchronous calling methods for batch training data generation.
"""

from abc import ABC, abstractmethod
from typing import Any

from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError, Timeout


class AbstractResponse(ABC):
    """
    Abstract base class for encapsulating interfaces, defining unified interface specifications
    
    Provides common configuration management and abstract methods, subclasses must implement the reply method.
    """
    
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        """
        Initialize the abstract response base class
        
        Args:
            base_url: API base URL
            api_key: API key
            model: Model name
        """
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.temperature: float = 0.7
        self.max_tokens: int = 2048
    
    @abstractmethod
    def reply(self, message: str) -> Any:
        """
        Abstract method, subclasses must implement
        
        Args:
            message: User message
            
        Returns:
            Model response content
        """
        pass
    
    def set_temperature(self, temperature: float) -> None:
        """
        Set temperature parameter
        
        Args:
            temperature: Temperature parameter, range [0, 2], higher values increase randomness
            
        Raises:
            ValueError: If temperature is out of range [0, 2]
        """
        if not 0 <= temperature <= 2:
            raise ValueError(f"Temperature {temperature} out of range [0, 2]")
        self.temperature = temperature
    
    def set_max_tokens(self, max_tokens: int) -> None:
        """
        Set maximum number of tokens to generate
        
        Args:
            max_tokens: Maximum number of tokens to generate
            
        Raises:
            ValueError: If max_tokens is not positive
        """
        if max_tokens <= 0:
            raise ValueError(f"Max tokens {max_tokens} must be positive")
        self.max_tokens = max_tokens


class Response(AbstractResponse):
    """
    Synchronous OpenAI interface implementation
    
    Uses the synchronous client from OpenAI SDK, providing a simple request-response pattern.
    Suitable for single or few request scenarios.
    """
    
    def __init__(self, *, base_url: str, api_key: str, model: str, **kwargs) -> None:
        """
        Initialize the synchronous response client
        
        Args:
            base_url: API base URL
            api_key: API key
            model: Model name
            **kwargs: Additional OpenAI client parameters (e.g., timeout, max_retries, etc.)
            
        Raises:
            Exception: If OpenAI client initialization fails
        """
        super().__init__(base_url, api_key, model)
        
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            **kwargs
        )
    
    def reply(self, message: str) -> str:
        """
        Synchronously call the API and return the response
        
        This method is used for model distillation, does not need to save long message history,
        just needs to provide a question and get a reply.
        
        Args:
            message: User message
            
        Returns:
            Model response text
            
        Raises:
            APIConnectionError: If API connection fails
            Timeout: If request times out
            RateLimitError: If rate limit is exceeded
            APIError: If API returns an error
            Exception: For any unexpected errors
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": message}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return response.choices[0].message.content
