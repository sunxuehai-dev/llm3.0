import requests
import json
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

BASE_URL = "http://localhost:8000/api/playground"
TEST_FILES_DIR = os.path.join(PROJECT_ROOT, "test_files")

def test_local_whisper_integration():
    print(f"\n{'='*60}")
    print("Testing Local Whisper Integration (Real API + Local File)")
    print(f"{'='*60}")

    # 1. Locate Test File
    # Using the file identified from file search
    audio_filename = "qwen3-omni-flash_Text_plus_Image_Input_20260127_163357.wav"
    audio_path = os.path.join(TEST_FILES_DIR, audio_filename)
    
    if not os.path.exists(audio_path):
        print(f"❌ Test file not found: {audio_path}")
        # Try finding any wav file in test_files as fallback
        found = False
        if os.path.exists(TEST_FILES_DIR):
            for f in os.listdir(TEST_FILES_DIR):
                if f.endswith(".wav"):
                    audio_path = os.path.join(TEST_FILES_DIR, f)
                    print(f"⚠️ Using fallback file: {audio_path}")
                    found = True
                    break
        
        if not found:
            print("❌ No suitable .wav file found for testing.")
            return

    print(f"📂 Input Audio: {audio_path}")

    # 2. Construct Payload
    payload = {
        "provider": "local",
        "model": "local-whisper-large",
        "input": audio_path, # Directly passing local absolute path
        "config": {
            "temperature": 0.0,
            # "language": "zh" # Optional: force Chinese
        }
    }

    print(f"\n[Request Payload]: {json.dumps(payload, ensure_ascii=False, indent=2)}")

    # 3. Send Request
    try:
        response = requests.post(f"{BASE_URL}/audio/transcription", json=payload)
        
        print(f"\n[Response Status]: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ [PASS] Transcription Successful")
            print(f"\n[Transcribed Text]:\n{result.get('text', '')}")
            print(f"\n[Raw Output]:\n{result.get('raw', '')}")
        else:
            print(f"❌ [FAIL] Error {response.status_code}: {response.text}")

    except Exception as e:
        print(f"❌ [FAIL] Exception: {e}")

if __name__ == "__main__":
    test_local_whisper_integration()
