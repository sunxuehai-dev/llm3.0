from typing import List, Optional, Union, Dict, Any
from typing_extensions import Literal
from pydantic import BaseModel, Field

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ModelConfig(BaseModel):
    temperature: Optional[float] = Field(default=0.85, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.8, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    repetition_penalty: Optional[float] = Field(default=None, ge=1.0)
    enable_search: Optional[bool] = Field(default=False)
    enable_thinking: Optional[bool] = Field(default=False) # Copilot/Deep Thinking
    max_tokens: Optional[int] = Field(default=2048, ge=1)
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    stream: bool = True
    # Omni Model Params (qwen3-omni-flash 语音输出)
    modalities: Optional[List[str]] = Field(default=None) # ["text"] or ["text", "audio"]
    audio: Optional[Dict[str, Any]] = Field(default=None) # {"voice": "Cherry", "format": "wav"}
    voice: Optional[str] = Field(default=None) # 音色: Cherry, Serena, Ethan, Chelsie

class ChatRequest(BaseModel):
    provider: Literal["aliyun", "deepseek", "doubao", "local"]
    model: str
    messages: List[Message]
    config: ModelConfig = Field(default_factory=ModelConfig)

class ChatResponseChunk(BaseModel):
    content: str
    reasoning_content: Optional[str] = None # For Thinking models
    finish_reason: Optional[str] = None
    audio: Optional[Dict[str, Any]] = None # {"data": "base64...", "format": "wav"}
