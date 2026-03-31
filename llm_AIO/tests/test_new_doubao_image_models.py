import requests
import json
import time
import os
import datetime
from typing import Dict, Any

# Get project root folder
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")
TEST_FILES_DIR = os.path.join(PROJECT_ROOT, "test_files")
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

# Local server must be running on this port
API_URL = "http://localhost:8000/api/playground/images/generations"
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

# The four new Seedream models to test
MODELS = [
    {
        "id": "doubao-seedream-5-0-260128",
        "description": "Seedream 5.0 Lite (综合旗舰)",
        "params": {
            "size": "2K",
            "watermark": True,
            "sequential_image_generation": "disabled"
        }
    },
    {
        "id": "doubao-seedream-4-5-251128",
        "description": "Seedream 4.5",
        "params": {
            "size": "2048x2048",
            "watermark": False
        }
    },
    {
        "id": "doubao-seedream-4-0-250828",
        "description": "Seedream 4.0",
        "params": {
            "size": "2048x2048",
            "sequential_image_generation": "auto"
        }
    },
    {
        "id": "doubao-seedream-3-0-t2i-250415",
        "description": "Seedream 3.0 (文生图专用)",
        "params": {
            "size": "1024x1024",
            "seed": 12345
        }
    }
]

def test_model(model_info: Dict[str, Any], image_url: str = None):
    model_id = model_info["id"]
    desc = model_info["description"]
    config = model_info["params"]
    
    print(f"\n======================================")
    print(f"Testing Image Model: {model_id} [{desc}]" + (" (Image-to-Image)" if image_url else " (Text-to-Image)"))
    print(f"Params sent from client: {config}")
    print(f"======================================")
    
    # Simulate frontend ImageGenRequest
    payload = {
        "provider": "doubao",
        "model": model_id,
        "prompt": "把参考图转换成赛博朋克风格" if image_url else "生化危机9里面的爱丽丝和女主角抱在一起在雨中穿梭的画面。",
        "config": config
    }
    
    if image_url:
        payload["image"] = image_url
    
    try:
        start_time = time.time()
        response = requests.post(
            API_URL, 
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120 # Image generation can take a while
        )
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                 print(f"❌ [FAIL] API returned an error: {data['error']}")
            elif "data" in data and len(data["data"]) > 0:
                 print(f"✅ [PASS] {model_id} - Generation Successful in {end_time - start_time:.2f}s")
                 print(f"Generated {len(data['data'])} image(s)")
                 for i, img in enumerate(data['data']):
                     print(f"Image {i+1} URL: {img.get('url', 'URL not found, might be base64')}")
                 
                 # Save output for URL review
                 timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                 result_filename = f"{model_id.replace('-', '_')}_output_{timestamp}.json"
                 result_path = os.path.join(TEST_RESULTS_DIR, result_filename)
                 with open(result_path, "w", encoding="utf-8") as f:
                     json.dump(data, f, ensure_ascii=False, indent=4)
                 print(f"📁 Detailed API response saved to {result_path}")
            else:
                 print(f"⚠️ [WARN] Response received but no valid image data found: {data}")
        else:
            print(f"❌ [FAIL] HTTP Error {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print(f"❌ [FAIL] {model_id} - Request Timed Out after 120s")
    except requests.exceptions.ConnectionError:
        print(f"❌ [FAIL] Could not connect to {API_URL}. Is the local server running?")
    except Exception as e:
        print(f"❌ [FAIL] Unexpected error: {str(e)}")


if __name__ == "__main__":
    print("Starting Doubao Image Generation Integration Tests...")
    print("Ensure that you have set VOLCENGINE_API_KEY in your environment and your custom backend is running on localhost:8000.")
    print("WARNING: This test makes direct API calls to Volcano Engine and may incur costs.")
    
    print("\n--- Pre-uploading Test Image ---")
    test_image_url = upload_file_to_oss("1212.png", "image")
    
    for i, model in enumerate(MODELS):
        # Text-to-Image
        test_model(model)
        time.sleep(2) # Prevent rate limiting
        
        # Image-to-Image only for the first 3 models
        if i < 3 and test_image_url:
            test_model(model, image_url=test_image_url)
            time.sleep(2)

    print("\n--- Testing Complete ---")
