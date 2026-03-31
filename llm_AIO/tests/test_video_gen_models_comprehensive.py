import requests
import json
import sys
import os
import time

# 获取项目根目录 (tests 的上一级)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")
TEST_FILES_DIR = os.path.join(PROJECT_ROOT, "test_files")

BASE_URL = "http://localhost:8000/api/playground"
UPLOAD_BASE_URL = "http://localhost:8000"

ONLINE_IMAGE_URL = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
LOCAL_IMAGE_NAME = "1212.png"

def upload_file_to_oss(file_path):
    print(f"📤 Uploading local image to OSS: {file_path}")
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'image/png')}
            data = {'user_id': 'video_test_user'}
            response = requests.post(
                f"{UPLOAD_BASE_URL}/api/image/upload",
                files=files,
                data=data,
                timeout=60
            )
        
        if response.status_code == 200:
            result = response.json()
            url = result.get("url")
            print(f"✅ Upload success! URL: {url}")
            return url
        else:
            print(f"❌ Upload failed (Status {response.status_code}): {response.text}")
            return None
    except Exception as e:
        print(f"❌ Upload exception: {e}")
        return None

def get_test_image_url():
    local_path = os.path.join(TEST_FILES_DIR, LOCAL_IMAGE_NAME)
    if os.path.exists(local_path):
        print(f"📂 Found local image: {local_path}")
        url = upload_file_to_oss(local_path)
        if url:
            return url
        print("⚠️ Upload failed, falling back to online URL")
    
    print("⚠️ Local image not found, using online URL")
    return ONLINE_IMAGE_URL

MODELS = [
    {
        "id": "wan2.6-i2v-flash",
        "description": "Flash version supports flexible duration (2-15) and audio",
        "params": {
            "resolution": "1280x720",
            "duration": 5, # Int
            "prompt_extend": True,
            "generate_audio": True
        }
    },
    # {
    #     "id": "wan2.6-i2v",
    #     "description": "Standard version supports fixed durations (5, 10, 15) and NO audio",
    #     "params": {
    #         "resolution": "1280x720", # 1920x1080 often fails on trial quota, safer with 720p
    #         "duration": 5,
    #         "prompt_extend": True,
    #         "generate_audio": False 
    #     }
    # }
]

def test_video_model(model_info, image_url):
    model_id = model_info["id"]
    params = model_info["params"]
    print(f"Testing Model: {model_id} with params: {params}")
    
    # Video Gen Request needs 'image_url' at top level for I2V
    payload = {
        "provider": "aliyun",
        "model": model_id,
        "prompt": "可以让这个图片动起来吗，动作幅度大一点", # Updated prompt
        "image_url": image_url,
        "config": params
    }
    
    # Print preview
    preview = json.dumps(payload, ensure_ascii=False)
    if len(preview) > 300: preview = preview[:300] + "...}"
    print(f"\n[Input Data]: {preview}")
    
    try:
        response = requests.post(f"{BASE_URL}/videos/generations", json=payload, timeout=120)

        if response.status_code == 200:
            data = response.json()
            
            # Check for API-level errors wrapped in success response
            if "error" in data:
                 print(f"❌ [FAIL] {model_id} - API Error: {data['error']}")
                 if "AccessDenied" in str(data['error']) or "403" in str(data['error']):
                     print("💡 提示: 请登录阿里云百炼控制台，开通 '通义万相-图生视频' 服务。")
                 return False
                 
            print(f"✅ [PASS] {model_id} - Response Received.")
            
            video_url = None
            if "output" in data and "video_url" in data["output"]:
                 video_url = data["output"]["video_url"]
            elif "output" in data and "url" in data["output"]:
                 video_url = data["output"]["url"]
            elif "data" in data and len(data["data"]) > 0:
                 video_url = data["data"][0].get("url")
            
            if video_url:
                import datetime
                os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(TEST_RESULTS_DIR, f"{model_id}_output_{timestamp}.mp4")
                print(f"⬇️ Downloading video from {video_url}...")
                try:
                    vid_data = requests.get(video_url).content
                    with open(filename, "wb") as f:
                        f.write(vid_data)
                    print(f"💾 Saved video to: {os.path.abspath(filename)}")
                except Exception as e:
                    print(f"⚠️ [WARN] Failed to download video: {e}")
            else:
                 print(f"⚠️ [WARN] No video URL found in immediate response (might be async task ID wrapper). Data: {data}")
                 # Check if the data implies an error we missed
                 if "code" in str(data) and "AccessDenied" in str(data):
                     print("💡 提示: 这是一个权限错误，请检查是否已开通服务。")
                     return False
                     
            return True
        else:
            print(f"❌ [FAIL] {model_id} - HTTP Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ [FAIL] {model_id} - Exception: {e}")
        return False

if __name__ == "__main__":
    # Force unset proxy variables for test execution
    import os
    for key in ["NO_PROXY", "no_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"]:
        if key in os.environ: del os.environ[key]

    print("--- Comprehensive Video Gen Model Test ---")
    
    # Resolve Image URL
    TEST_IMAGE_URL = get_test_image_url()
    
    results = {}
    for m in MODELS:
        success = test_video_model(m, TEST_IMAGE_URL)
        results[m["id"]] = "PASS" if success else "FAIL"
    
    print("\n--- Summary ---")
    for mid, res in results.items():
        status = "✅" if res == "PASS" else "❌"
        print(f"{status} {mid}: {res}")

    # ==========================================
    # 🆕 T2V 测试 (Text-to-Video Test)
    # ==========================================
    print("\n" + "="*40)
    print("🧪 [NEW] Testing Text-to-Video (No Image)")
    print("="*40)
    
    t2v_model_id = "wan2.6-i2v-flash"
    print(f"Testing Model: {t2v_model_id} (Text Only Mode)")
    
    t2v_payload = {
        "provider": "aliyun",
        "model": t2v_model_id,
        "prompt": "一只奔跑的小金毛，高清，3d风格", 
        # image_url is intentionally OMITTED or None
        "config": {
            "resolution": "1280x720",
            "duration": 5,
            "generate_audio": True
        }
    }
    
    print(f"[Input Data]: {json.dumps(t2v_payload, ensure_ascii=False)}")
    
    try:
        t2v_response = requests.post(f"{BASE_URL}/videos/generations", json=t2v_payload, timeout=120)
        
        if t2v_response.status_code == 200:
             print(f"✅ [PASS] Text-to-Video Request Accepted.")
             data = t2v_response.json()
             if "output" in data and "video_url" in data["output"]:
                 print(f"   🎥 Video URL: {data['output']['video_url']}")
             else:
                 print(f"   ⚠️ Response: {data}")
        elif t2v_response.status_code == 422:
             print(f"❌ [FAIL] Validation Error (Schema Fix Failed?): {t2v_response.text}")
        else:
             print(f"❌ [FAIL] HTTP {t2v_response.status_code}: {t2v_response.text}")
            
    except Exception as e:
        print(f"❌ [FAIL] Exception: {e}")
