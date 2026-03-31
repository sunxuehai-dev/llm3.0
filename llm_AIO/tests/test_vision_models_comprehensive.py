import requests
import json
import sys
import os

# 获取项目根目录 (tests 的上一级)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_URL = "http://localhost:8000/api/playground"
UPLOAD_BASE_URL = "http://localhost:8000"

# ==========================================
# 📷 用户设置 (User Settings)
# ==========================================
# 方式 1: 测试在线图片 URL (将下方 URL 替换为您想测试的链接)
# 示例: "http://your-domain.com/image.jpg"
ONLINE_IMAGE_URL = "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"

# 方式 2: 测试本地图片 (如果有值，则优先上传到阿里云 OSS 获取 URL)
# 将其设置为 None 以强制使用上面的 ONLINE_IMAGE_URL
LOCAL_IMAGE_PATH = os.path.join(PROJECT_ROOT, "test_files", "1212.png")
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")

MODELS = [
    {
        "id": "qwen-vl-max",
        "params": {
            "temperature": 0.7,
            "top_p": 0.8
            # enable_thinking NOT supported
        }
    },
    {
        "id": "qwen-vl-plus",
        "params": {
            "temperature": 0.5,
            "top_p": 0.5
        }
    },
    {
        "id": "qwen3-vl-plus",
        "params": {
            "temperature": 0.8,
            "top_p": 0.9,
            "enable_thinking": True
        }
    },
    {
        "id": "qwen3-vl-flash",
        "params": {
            "temperature": 0.1,
            "top_p": 0.1,
            "enable_thinking": False
        }
    }
]

def upload_local_image_to_oss(local_path):
    """
    将本地图片上传到阿里云 OSS，返回可访问的 URL
    """
    print(f"📤 正在上传本地图片到阿里云 OSS: {local_path}")
    
    if not os.path.exists(local_path):
        print(f"❌ 本地图片不存在: {local_path}")
        return None
    
    try:
        with open(local_path, 'rb') as f:
            files = {'file': (os.path.basename(local_path), f, 'image/png')}
            data = {'user_id': 'vision_test_user'}
            
            response = requests.post(
                f"{UPLOAD_BASE_URL}/api/image/upload",
                files=files,
                data=data,
                timeout=60
            )
        
        if response.status_code == 200:
            result = response.json()
            url = result.get("url")
            image_id = result.get("image_id")
            print(f"✅ 上传成功!")
            print(f"   📎 图片ID: {image_id}")
            print(f"   🔗 OSS URL: {url}")
            return url
        else:
            print(f"❌ 上传失败 (状态码 {response.status_code}): {response.text}")
            return None
    except Exception as e:
        print(f"❌ 上传异常: {e}")
        return None

def test_vision_model(model_info, image_url):
    model_id = model_info["id"]
    params = model_info["params"]
    print(f"Testing Model: {model_id} with params: {params}")
    print(f"[INFO] Using Image URL: {image_url}")

    # Construct Multimodal Message (Dashscope Format)
    content = [
        {"image": image_url},
        {"text": "请详细描述这张图片。"}
    ]
    
    payload = {
        "provider": "aliyun",
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "config": params
    }
    print(f"\n[Input Data]: {json.dumps(payload, ensure_ascii=False)}")

    
    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload, stream=True)
        if response.status_code == 200:
            print(f"✅ [PASS] {model_id} - Connection Established.")
            print(f"✅ [PASS] {model_id} - Connection Established.")
            # Consume stream
            full_content = ""
            error_detected = False
            import datetime
            os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            print(f"[Receiving Full Output]...")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if "Internal Error" in decoded_line or "Exception" in decoded_line or "Error" in decoded_line:
                        error_detected = True
                        # Don't print every error line, just flag it
                        
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            content_piece = ""
                            
                            # 1. Standard OpenAI Format
                            if "choices" in data_json and len(data_json["choices"]) > 0:
                                delta = data_json["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content_piece = delta["content"]
                            
                            # 2. Legacy Flat Format (Fallback)
                            elif "content" in data_json:
                                content_piece = data_json["content"]
                                
                            if content_piece:
                                full_content += content_piece
                        except:
                            pass

            print(f"\n[Full Response Start] --------------------------------")
            print(full_content)
            print(f"[Full Response End] ----------------------------------")
            
            filename = os.path.join(TEST_RESULTS_DIR, f"{model_id}_output_{timestamp}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(full_content)
            print(f"💾 Saved full response to: {os.path.abspath(filename)}")

            # Check for errors in response content
            if "Error" in full_content or "AccessDenied" in full_content or "Failed to download" in full_content:
                print(f"❌ [FAIL] {model_id} - Error detected in response content.")
                return False
            elif len(full_content.strip()) < 10:
                print(f"❌ [FAIL] {model_id} - Response too short or empty.")
                return False
            else:
                print(f"✅ [PASS] {model_id} - Stream completed successfully.")
                return True
        else:
            print(f"❌ [FAIL] {model_id} - Error {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ [FAIL] {model_id} - Exception: {e}")
        return False

if __name__ == "__main__":
    # Force unset proxy variables for test execution
    import os
    for key in ["NO_PROXY", "no_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"]:
        if key in os.environ: del os.environ[key]

    print("--- Comprehensive Vision Model Test ---")
    print()
    
    # Step 1: Determine Image URL
    image_url_to_use = None
    
    if LOCAL_IMAGE_PATH and os.path.exists(LOCAL_IMAGE_PATH):
        print(f"📷 发现本地图片: {LOCAL_IMAGE_PATH}")
        print(f"🔄 将上传到阿里云 OSS 以获取可访问的 URL...")
        print()
        
        # Upload local image to OSS
        uploaded_url = upload_local_image_to_oss(LOCAL_IMAGE_PATH)
        
        if uploaded_url:
            image_url_to_use = uploaded_url
            print()
        else:
            print(f"⚠️ 本地图片上传失败，回退到使用在线图片 URL")
            image_url_to_use = ONLINE_IMAGE_URL
    else:
        print(f"📷 未找到本地图片 ({LOCAL_IMAGE_PATH})，使用在线图片 URL")
        image_url_to_use = ONLINE_IMAGE_URL
    
    print(f"🖼️ 最终使用的图片 URL: {image_url_to_use}")
    print()
    print("-" * 60)
    print()
    
    # Step 2: Test all models with the same image URL
    results = {}
    for m in MODELS:
        success = test_vision_model(m, image_url_to_use)
        results[m["id"]] = "PASS" if success else "FAIL"
        print()
    
    print("--- Summary ---")
    for mid, res in results.items():
        status_icon = "✅" if res == "PASS" else "❌"
        print(f"{status_icon} {mid}: {res}")

    # ==========================================
    # 🆕 BASE64 测试 (Base64 Test)
    # ==========================================
    print("\n" + "="*40)
    print("🧪 [NEW] Testing Base64 Auto-Conversion")
    print("="*40)
    
    if LOCAL_IMAGE_PATH and os.path.exists(LOCAL_IMAGE_PATH):
        try:
            import base64
            with open(LOCAL_IMAGE_PATH, "rb") as img_file:
                b64_data = base64.b64encode(img_file.read()).decode('utf-8')
                
            b64_string = f"data:image/png;base64,{b64_data}"
            print(f"📦 Encoded local image to Base64 (Length: {len(b64_string)})")
            
            # Send Base64 directly
            # Picking qwen-vl-max for this test
            b64_model_id = "qwen-vl-max"
            b64_payload = {
                "provider": "aliyun",
                "model": b64_model_id,
                "messages": [{
                    "role": "user", 
                    "content": [
                        {"image": b64_string}, # Passing Base64 directly
                        {"text": "[Base64 Test] 这张图片里有什么？"}
                    ]
                }],
                "config": {"temperature": 0.1}
            }
            
            print(f"🚀 Sending Base64 request to {b64_model_id}...")
            # We don't print the full payload to avoid spamming the console with base64
            
            b64_response = requests.post(f"{BASE_URL}/chat", json=b64_payload, stream=True)
            
            if b64_response.status_code == 200:
                print(f"✅ [PASS] Base64 Request Accepted (200 OK)")
                print("[Output]: ", end="")
                for line in b64_response.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith("data: "):
                            try:
                                json_d = json.loads(decoded[6:])
                                if "choices" in json_d:
                                     print(json_d["choices"][0]["delta"].get("content", ""), end="")
                            except: pass
                print("\n✅ Base64 Test Completed Successfully!")
            else:
                print(f"❌ [FAIL] Base64 Request Failed: {b64_response.status_code} - {b64_response.text}")
                
        except Exception as e:
            print(f"❌ [FAIL] Base64 Test Exception: {e}")
    else:
        print("⚠️ Skipping Base64 test (No local image found)")
