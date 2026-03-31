from typing import List, Optional, Dict, Any
from typing_extensions import Literal
from pydantic import BaseModel, Field

# --- Base Config ---
class BaseConfig(BaseModel):
    pass

# --- 1. Audio (TTS) Models ---
class AudioSpeechConfig(BaseModel):
    speed: float = 1.0
    voice: Optional[str] = "default"
    volume: Optional[int] = 50
    pitch: float = 1.0
    format: str = "mp3"
    sample_rate: int = 16000
    language_type: Optional[str] = "Auto"

class AudioSpeechRequest(BaseModel):
    provider: str
    model: str
    input: str # Text to speak
    config: AudioSpeechConfig = Field(default_factory=AudioSpeechConfig)

# --- 2. Audio (ASR) Models ---
class AudioTranscriptionConfig(BaseModel):
    format: str = "mp3"
    sample_rate: int = 16000
    enable_punctuation_prediction: bool = True
    enable_inverse_text_normalization: bool = True
    disfluency_removal_enabled: bool = False
    speech_noise_threshold: Optional[float] = None
    # New Params
    speaker_diarization_enabled: bool = False # For fun-asr / paraformer
    enable_semantic_break: bool = False # For qwen-asr
    # Whisper / Local Params
    temperature: Optional[float] = 0.0
    language: Optional[str] = None

class AudioTranscriptionRequest(BaseModel):
    provider: str
    model: str
    input: str 
    # For ASR, input is usually a URL or base64 encoded string, or we might handle file upload separately in the route.
    # But for the adapter interface, let's assume 'input' is the file path or URL.
    config: AudioTranscriptionConfig = Field(default_factory=AudioTranscriptionConfig)

# --- 2. Image Generation Models ---
class ImageGenConfig(BaseModel):
    size: str = "1024x1024"
    steps: Optional[int] = 30 # For local models
    negative_prompt: Optional[str] = None
    prompt_extend: Optional[bool] = True
    # Seedream / DALL-E Extended Params
    seed: Optional[int] = None
    sequential_image_generation: Optional[str] = None # auto / disabled
    stream: Optional[bool] = None
    watermark: Optional[bool] = None
    response_format: Optional[str] = None

class ImageGenRequest(BaseModel):
    provider: str
    model: str
    prompt: str
    image: Optional[Any] = None # Support single or multiple images for i2i / i2t2i
    config: ImageGenConfig = Field(default_factory=ImageGenConfig)

# --- 3. Image Analysis (Vision) Models ---
# Can share ChatRequest structure but with image inputs in content

# --- 4. Video Generation Models ---
# --- 4. Video Generation Models ---
class VideoGenConfig(BaseModel):
    resolution: str = "1280x720" # 720P, 1080P
    duration: int = 5
    prompt_extend: bool = True
    shot_type: Optional[str] = "single" # single, multi
    generate_audio: bool = True # Maps to 'audio' param
    
    # Doubao Seedance Specifics
    ratio: Optional[str] = None
    seed: Optional[int] = None
    camera_fixed: Optional[bool] = None
    watermark: Optional[bool] = None

class VideoGenRequest(BaseModel):
    provider: str
    model: str
    prompt: Optional[str] = None # Optional if template is used
    negative_prompt: Optional[str] = None
    image_url: Optional[str] = None # Optional for T2V models, Required for I2V models (by Aliyun API)
    audio_url: Optional[str] = None
    template: Optional[str] = None
    config: VideoGenConfig = Field(default_factory=VideoGenConfig)
