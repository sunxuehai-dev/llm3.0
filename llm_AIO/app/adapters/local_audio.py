import os
import tempfile
import aiohttp
import asyncio
from typing import Dict, Any
from app.models.multimodal import AudioTranscriptionRequest

class LocalAudioAdapter:
    def __init__(self):
        self.model = None
        self.model_name = "large" # Default to large, configurable

    def _load_model(self):
        # Configure FFmpeg path manually for Windows
        ffmpeg_path = r"C:\Users\aiahu\Desktop" # Directory containing ffmpeg.exe
        if os.path.exists(ffmpeg_path) and ffmpeg_path not in os.environ["PATH"]:
            print(f"Adding FFmpeg to PATH: {ffmpeg_path}")
            os.environ["PATH"] += os.pathsep + ffmpeg_path

        if self.model is None:
            try:
                import whisper
                import torch
            except ImportError:
                return {"error": "Missing dependency: openai-whisper. Please install it with `pip install openai-whisper`."}
            
            try:
                # 1. Define custom download root to be explicit
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                download_root = os.path.join(project_root, "models", "whisper")
                os.makedirs(download_root, exist_ok=True)
                
                # 2. Check if model already exists (heuristic based on Whisper's naming)
                # whisper.load_model handles caching, but we want to log it explicitely for the user
                print(f"Checking for Whisper model '{self.model_name}' in {download_root}...")
                
                # 3. Load model (Whisper uses tqdm internally for download progress if not present)
                # We force download_root so we know where it is
                print(f"Loading/Downloading Whisper model '{self.model_name}'...")
                print(f"Note: If this is the first run, it will download approx 1.5GB-3GB. Please wait...")
                
                self.model = whisper.load_model(self.model_name, download_root=download_root)
                print(f"✅ Whisper model '{self.model_name}' loaded successfully.")
                
            except Exception as e:
                return {"error": f"Failed to load Whisper model: {str(e)}"}
        return None

    async def transcribe(self, request: AudioTranscriptionRequest) -> Dict[str, Any]:
        """
        Transcribe audio using local Whisper model.
        """
        # Ensure model is ready
        error = self._load_model()
        if error:
            return error

        input_url = request.input
        file_path = None
        temp_file = None

        try:
            # 1. Resolve Input to Local Path
            if os.path.exists(input_url):
                file_path = input_url
            elif input_url.startswith("http"):
                # Download remote file/local server URL to temp file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{request.config.format or 'wav'}")
                file_path = temp_file.name
                temp_file.close() # Close to allow writing/reading by valid process

                async with aiohttp.ClientSession() as session:
                    async with session.get(input_url) as resp:
                        if resp.status != 200:
                            return {"error": f"Failed to download audio from {input_url}: {resp.status}"}
                        with open(file_path, 'wb') as f:
                            f.write(await resp.read())
            else:
                 return {"error": "Invalid input: Must be a valid URL or local file path."}

            # 2. Run Whisper Transcription (in executor to avoid blocking)
            # Whisper parameters
            options = {
                "temperature": request.config.temperature if request.config.temperature is not None else 0.0,
                # "language": request.config.language, # Add if supported in request config
            }
            
            # Run in thread pool because whisper is CPU/GPU intensive and synchronous
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: self.model.transcribe(file_path, **options))

            return {
                "text": result.get("text", "").strip(),
                "raw": result # Return full result including segments if needed
            }

        except Exception as e:
            return {"error": f"Transcription failed: {str(e)}"}
            
        finally:
            # Cleanup temp file
            if temp_file and os.path.exists(file_path):
                try:
                    os.unlink(file_path)
                except:
                    pass
