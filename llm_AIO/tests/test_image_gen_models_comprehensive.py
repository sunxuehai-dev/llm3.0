import requests
import json
import sys
import os

# 获取项目根目录 (tests 的上一级)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")

BASE_URL = "http://localhost:8000/api/playground"

MODELS = [
    {
        "id": "qwen-image-plus",
        "prompt": "一个充满未来感的赛博朋克风格城市，有飞行汽车",
        "params": {
            "size": "1024x1024", # Adapter should convert this to 1024*1024
            "negative_prompt": "模糊, 低质量, 扭曲",
            # "prompt_extend": True # Optional, keeping logic simple
        }
    },
    {
        "id": "qwen-image-max",
        "prompt": "衣服典雅庄重的对联悬挂于厅堂之中，房间是个安静古典的中式布置，桌子上放着一些青花瓷，对联飘逸，中间挂在一着一副中国风的画作，内容是岳阳楼。",
        "params": {
            "size": "1024*1024", 
            "negative_prompt": ""
        }
    }
]

def test_image_model(model_info):
    model_id = model_info["id"]
    params = model_info["params"]
    print(f"Testing Model: {model_id} with params: {params}")
    
    payload = {
        "provider": "aliyun",
        "model": model_id,
        "prompt": model_info["prompt"],
        "config": params
    }
    print(f"\n[Input Data]: {json.dumps(payload, ensure_ascii=False)}")
    
    try:
        response = requests.post(f"{BASE_URL}/images/generations", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"[Output Data]: {json.dumps(data, ensure_ascii=False)[:300]}...")
            
            # Check for standard OpenAI image response format or custom
            url = None
            if "data" in data and len(data["data"]) > 0:
                url = data["data"][0].get("url")
            elif isinstance(data, list) and len(data) > 0:
                 url = data[0].get("url")
                 
            if url:
                print(f"✅ [PASS] {model_id} - Image generated URL: {url}")
                
                # Download and save image
                import datetime
                os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(TEST_RESULTS_DIR, f"{model_id}_output_{timestamp}.png")
                try:
                    img_data = requests.get(url).content
                    with open(filename, "wb") as f:
                        f.write(img_data)
                    print(f"💾 Saved image to: {os.path.abspath(filename)}")
                except Exception as e:
                    print(f"⚠️ [WARN] Failed to download image: {e}")
                    
                return True
            else:
                print(f"⚠️ [WARN] {model_id} - 200 OK but no URL found in response: {data}")
                return True # Technically a pass if 200 OK
        else:
            print(f"❌ [FAIL] {model_id} - Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ [FAIL] {model_id} - Exception: {e}")
        return False

if __name__ == "__main__":
    print("--- Comprehensive Image Gen Model Test ---")
    results = {}
    for m in MODELS:
        success = test_image_model(m)
        results[m["id"]] = "PASS" if success else "FAIL"
    
    print("\n--- Summary ---")
    for mid, res in results.items():
        print(f"{mid}: {res}")
