import requests
import json
import time
import os

BASE_URL = "http://localhost:8000/api/playground/videos/generations"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")

def test_t2v_model(model_name, payload, description):
    print(f"\n" + "="*50)
    print(f"🧪 Testing {model_name}: {description}")
    print(f"="*50)
    print(f"[Request Payload]:\n{json.dumps(payload, ensure_ascii=False, indent=2)}")
    
    try:
        response = requests.post(BASE_URL, json=payload, timeout=300) # Increased timeout to 5 mins
        
        if response.status_code == 200:
            data = response.json()
            if "output" in data and "video_url" in data["output"]:
                video_url = data["output"]["video_url"]
                print(f"\n✅ [PASS] Video Generated Successfully!")
                print(f"   🎥 URL: {video_url}")
                
                # Download
                os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
                timestamp = int(time.time())
                filename = os.path.join(TEST_RESULTS_DIR, f"{model_name}_{timestamp}.mp4")
                print(f"   ⬇️ Downloading to {filename}...")
                try:
                    with open(filename, "wb") as f:
                        f.write(requests.get(video_url).content)
                    print("   💾 Download Complete.")
                except Exception as e:
                    print(f"   ⚠️ Download Failed: {e}")
                
            elif "error" in data:
                 print(f"\n❌ [FAIL] API Returned Error: {data['error']}")
            else:
                 print(f"\n⚠️ [WARN] Unexpected Response Format: {data}")
        else:
            print(f"\n❌ [FAIL] HTTP Error {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"\n❌ [FAIL] Exception/Network Error: {e}")

if __name__ == "__main__":
    print("=== Testing New T2V Models (All Parameters) ===")
    
    # 1. Test Wan2.6 T2V (Full Feature Set)
    # Supported: resolution, duration, prompt_extend, generate_audio, shot_type
    test_t2v_model(
        "wan2.6-t2v", 
        {
            "provider": "aliyun",
            "model": "wan2.6-t2v",
            "prompt": "一段史诗感的电影片头，展示古代中国山水画变活了，有仙鹤飞过，云雾缭绕。镜头缓慢推近。",
            "config": {
                "resolution": "1280x720",      # 支持 720P/1080P
                "duration": 5,                 # 支持 5/10/15
                "prompt_extend": True,        # 支持
                "generate_audio": True,       # 支持
                "shot_type": "multi"          # ✅ 独有参数：支持 single/multi
            }
        },
        "Verifying ALL params: Multi-shot, Audio, PromptExtend, 720P"
    )
    
    # 2. Test Wan2.5 T2V Preview (Standard Feature Set)
    # Supported: resolution, duration, prompt_extend, generate_audio
    # NOT Supported: shot_type
    test_t2v_model(
        "wan2.5-t2v-preview", 
        {
            "provider": "aliyun",
            "model": "wan2.5-t2v-preview",
            "prompt": "一只可爱的小猫在草地上打滚，阳光明媚，像素风格。",
            "config": {
                "resolution": "832x480",       # ✅ 特有测试：480P (16:9)
                "duration": 5,                 # 支持 5/10
                "prompt_extend": True,        # 支持
                "generate_audio": True,       # 支持
                # "shot_type": "multi"        # ❌ EXCLUDED (Not supported)
            }
        },
        "Verifying ALL params: 480P, Audio, PromptExtend (No ShotType)"
    )
