import requests
import json
import os
import datetime

# 获取项目根目录 (tests 的上一级)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")

BASE_URL = "http://localhost:8000/api/playground/audio"

# --- Test Configurations ---
TTS_TEST_CASES = [
    {
        "model": "cosyvoice-v3-flash",
        "description": "CosyVoice v3 Flash - Standard",
        "input": "大家好，我是这个项目的智能助手。很高兴为您服务。",
        "config": {
            "voice": "longanyang",
            "speed": 1.0,
            "volume": 50,
            "format": "mp3"
        }
    },
    {
        "model": "cosyvoice-v3-plus",
        "description": "CosyVoice v3 Plus - High Quality",
        "input": "这是一款音质极佳的语音合成模型，适合品牌定制和个性化克隆。",
        "config": {
            "voice": "longanyang",
            "speed": 1.0,
            "volume": 50,
            "format": "wav"
        }
    },
    {
        "model": "cosyvoice-v2",
        "description": "CosyVoice v2 - Dialect support",
        "input": "你好，我而家同你讲紧广度话。你听唔听得明啊？",
        "config": {
            "voice": "longyingxiao", 
            "speed": 1.0,
            "format": "mp3"
        }
    },

]

def test_audio_speech(case):
    model = case["model"]
    desc = case["description"]
    text = case["input"]
    config = case["config"]
    
    print(f"\n--- Testing TTS: {model} [{desc}] ---")
    
    payload = {
        "provider": "aliyun",
        "model": model,
        "input": text,
        "config": config
    }
    
    try:
        response = requests.post(f"{BASE_URL}/speech", json=payload)
        
        if response.status_code == 200:
            print(f"✅ [PASS] {model} - Response 200 OK")
            
            # Save audio file
            os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_ext = config.get("format", "mp3")
            filename = f"tts_{model}_{timestamp}.{file_ext}"
            filepath = os.path.join(TEST_RESULTS_DIR, filename)
            
            with open(filepath, "wb") as f:
                f.write(response.content)
            
            print(f"💾 Audio saved to: {os.path.abspath(filepath)}")
            return True
        else:
            print(f"❌ [FAIL] {model} - Status {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ [FAIL] {model} - Exception: {e}")
        return False

if __name__ == "__main__":
    print("=== Aliyun TTS Comprehensive Test Suite ===")
    
    results = {}
    for i, case in enumerate(TTS_TEST_CASES):
        success = test_audio_speech(case)
        results[f"Case_{i+1}_{case['model']}"] = "PASS" if success else "FAIL"
    
    print("\n" + "="*40)
    print("Summary of Aliyun TTS Tests")
    print("="*40)
    for k, res in results.items():
        status_icon = "✅" if res == "PASS" else "❌"
        print(f"{status_icon} {k}: {res}")
