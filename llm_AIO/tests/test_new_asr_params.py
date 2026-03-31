import requests
import json
import sys
import os

# Project root setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")
TEST_FILES_DIR = os.path.join(PROJECT_ROOT, "test_files")

BASE_URL = "http://localhost:8000/api/playground"

# Default test file
DEFAULT_AUDIO_FILE = os.path.join(TEST_FILES_DIR, "qwen3-omni-flash_Text_plus_Image_Input_20260127_163357.wav")

# Models to test with new parameters
MODELS_TO_TEST = [
    {
        "id": "qwen3-asr-flash-filetrans",  # New model requested
        "desc": "Qwen3 ASR File Trans (Basic)",
        "params": {
            "format": "wav",
            "sample_rate": 16000,
            "enable_punctuation_prediction": True,
            "enable_inverse_text_normalization": True,
            "disfluency_removal_enabled": False, # Test Parameter 1
            "enable_semantic_break": True # Qwen3 specific
        }
    },
    {
        "id": "qwen3-asr-flash-filetrans",
        "desc": "Qwen3 ASR File Trans (With Disfluency Removal)",
        "params": {
            "format": "wav",
            "sample_rate": 16000,
            "disfluency_removal_enabled": True, # Test Parameter 1: ON
        }
    },
    {
        "id": "fun-asr", # Assuming this maps to Paraformer or similar in implementation
        "desc": "FunASR / Paraformer (Speaker Diarization)",
        "params": {
            "format": "wav",
            "sample_rate": 16000,
            "disfluency_removal_enabled": True,
            "speaker_diarization_enabled": True, # Test Parameter 2: ON
            "max_speaker_count": 2
        }
    }
]

def test_asr_model(model_config, audio_path):
    model_id = model_config["id"]
    description = model_config.get("desc", model_id)
    params = model_config["params"]
    
    print(f"\n[{description}]")
    print(f"Model ID: {model_id}")
    print(f"Params: {json.dumps(params, ensure_ascii=False)}")
    
    # payload
    payload = {
        "provider": "aliyun", # Assuming aliyun provider for both
        "model": model_id,
        "input": audio_path, # Path will be handled by backend if local, or URL
        "config": params
    }
    
    try:
        url = f"{BASE_URL}/audio/transcription"
        print(f"POST {url}")
        
        # If input is a local path that the server can access (same machine), sending path string works.
        # But if server is remote, we'd need upload.
        # Here we assume local test environment where server can read the path.
        if not os.path.exists(audio_path) and not audio_path.startswith("http"):
             print(f"❌ File not found: {audio_path}")
             return False

        # In the existing router implementation, 'input' can be a path string.
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            data = response.json()
            text = data.get("text", "")
            raw = data.get("raw", "")
            
            print(f"✅ Success!")
            print(f"Transcribed Text Preview: {text[:200]}..." if len(text) > 200 else f"Transcribed Text: {text}")
            
            # Check if parameters had effect (manual verification usually needed, but we can check raw response for keys)
            # For speaker diarization, raw output usually contains 'sentences' with 'speaker_id'
            if params.get("speaker_diarization_enabled"):
                if "speaker_id" in str(raw) or "spk" in str(raw):
                    print("   -> Speaker Diarization seems PRESENT in raw output.")
                else:
                    print("   -> ⚠️ Speaker Diarization expected but might NOT be in raw output.")
            
            return True
        else:
            print(f"❌ Failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test New ASR Models & Params")
    parser.add_argument("audio_file", nargs="?", default=DEFAULT_AUDIO_FILE, help="Path to audio file")
    args = parser.parse_args()
    
    audio_file = args.audio_file
    if not os.path.exists(audio_file) and not audio_file.startswith("http"):
        print(f"Error: Default audio file not found at {audio_file}")
        print("Please provide a valid audio file path.")
        if os.path.exists("test_audio.mp3"):
            audio_file = os.path.abspath("test_audio.mp3")
            print(f"Using fallback: {audio_file}")
        else:
            sys.exit(1)
            
    print(f"Testing with audio: {audio_file}")
    
    for config in MODELS_TO_TEST:
        test_asr_model(config, audio_file)
