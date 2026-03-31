import asyncio
import os
import sys
import json
from dotenv import load_dotenv

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

from app.adapters.doubao import DoubaoAdapter
from app.models.multimodal import VideoGenRequest, VideoGenConfig

# Configuration
API_KEY = os.getenv("VOLCENGINE_API_KEY") or os.getenv("ARK_API_KEY")

MODELS_TO_TEST = [
    {"id": "doubao-seedance-1-5-pro-251215", "type": "both"},
    {"id": "doubao-seedance-1-0-pro-250528", "type": "both"},
    {"id": "doubao-seedance-1-0-pro-fast-251015", "type": "both"},
    {"id": "doubao-seedance-1-0-lite-i2v-250428", "type": "i2v_only"},
    {"id": "doubao-seedance-1-0-lite-t2v-250428", "type": "t2v_only"}
]

# --- File Upload Utility for Multimodal Testing ---
import requests
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_FILES_DIR = os.path.join(PROJECT_ROOT, "test_files")
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")
UPLOAD_BASE_URL = "http://localhost:8000"

def upload_file_to_oss(filename, file_type):
    file_path = os.path.join(TEST_FILES_DIR, filename)
    print(f"📤 Uploading local {file_type} to OSS: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        mime_map = {'image': 'image/png', 'audio': 'audio/wav', 'video': 'video/mp4'}
        mime_type = mime_map.get(file_type, 'application/octet-stream')
        
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, mime_type)}
            data = {'user_id': 'doubao_test_user'}
            
            response = requests.post(
                f"{UPLOAD_BASE_URL}/api/{file_type}/upload",
                files=files,
                data=data,
                timeout=120
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

async def test_doubao_video_gen():
    print("=== Testing Doubao Video Generation Adapter ===")
    
    if not API_KEY:
        print("Warning: VOLCENGINE_API_KEY or ARK_API_KEY environment variable not found.")
        print("Please ensure it is set in .env or environment.")
    else:
        print(f"Using API Key: {API_KEY[:4]}...{API_KEY[-4:]}")

    try:
        adapter = DoubaoAdapter()
    except Exception as e:
        print(f"Failed to initialize adapter: {e}")
        return
        
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

    # Pre-upload image for I2V tests
    print("\n--- Pre-uploading Test Image ---")
    image_url = upload_file_to_oss("1212.png", "image")

    for model_info in MODELS_TO_TEST:
        model_id = model_info["id"]
        test_type = model_info["type"]
        print(f"\n==================================================")
        print(f"🚀 Testing Model: {model_id}")
        print(f"==================================================")

        # Test 1: Text to Video
        if test_type in ["both", "t2v_only"]:
            print(f"\n--- [T2V] Text to Video ---")
            request_t2v = VideoGenRequest(
                provider="doubao",
                model=model_id,
                prompt="一只科幻赛博朋克风格的机器猫穿梭在城市屋顶，4K高画质，电影级质感",
                config=VideoGenConfig(
                    duration=5,
                    resolution="720p",
                    ratio="16:9",
                    watermark=True
                )
            )
            print("\n⏳ Submitting T2V Task...")
            t2v_result = await adapter.generate_video(request_t2v)
            if "output" in t2v_result and "video_url" in t2v_result["output"]:
                video_url = t2v_result["output"]["video_url"]
                print(f"✅ T2V Task Completed! URL: {video_url}")
                log_file = os.path.join(TEST_RESULTS_DIR, f"{model_id}_t2v_result.txt")
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(f"URL: {video_url}\n")
            else:
                print(f"❌ T2V Task Failed: {json.dumps(t2v_result, indent=2, ensure_ascii=False)}")

        # Test 2: Image to Video
        if test_type in ["both", "i2v_only"]:
            print(f"\n--- [I2V] Image to Video ---")
            if image_url:
                request_i2v = VideoGenRequest(
                    provider="doubao",
                    model=model_id,
                    prompt="让图片生动起来，展示出电影级的运镜感",
                    image_url=image_url,
                    config=VideoGenConfig(
                        duration=4,
                        resolution="720p",
                        ratio="16:9",
                        camera_fixed=False
                    )
                )
                print("\n⏳ Submitting I2V Task...")
                i2v_result = await adapter.generate_video(request_i2v)
                if "output" in i2v_result and "video_url" in i2v_result["output"]:
                    video_url = i2v_result["output"]["video_url"]
                    print(f"✅ I2V Task Completed! URL: {video_url}")
                    log_file = os.path.join(TEST_RESULTS_DIR, f"{model_id}_i2v_result.txt")
                    with open(log_file, "w", encoding="utf-8") as f:
                        f.write(f"URL: {video_url}\n")
                else:
                    print(f"❌ I2V Task Failed: {json.dumps(i2v_result, indent=2, ensure_ascii=False)}")
            else:
                print("Skipping I2V test because file upload failed.")

if __name__ == "__main__":
    asyncio.run(test_doubao_video_gen())
