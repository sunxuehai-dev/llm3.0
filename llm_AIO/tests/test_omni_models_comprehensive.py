import requests
import json
import socket
import os
import base64
import datetime

# 获取项目根目录 (tests 的上一级)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")
TEST_FILES_DIR = os.path.join(PROJECT_ROOT, "test_files")

# Check if port 8001 is open
BASE_URL = "http://localhost:8000/api/playground"
UPLOAD_BASE_URL = "http://localhost:8000"

# --- Test Configurations ---
ONLINE_IMAGE_URL = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"
ONLINE_AUDIO_URL = "https://dashscope.oss-cn-beijing.aliyuncs.com/samples/audio/paraformer/hello_world.wav"
ONLINE_VIDEO_URL = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg" # Placeholder

# Local file names in "test_files" directory
LOCAL_IMAGE_NAME = "1212.png"
LOCAL_AUDIO_NAME = "qwen3-omni-flash_Text_plus_Image_Input_20260127_163357.wav"
LOCAL_VIDEO_NAME = "wan2.6-i2v_output_20260127_162717.mp4" # Using existing video file

def upload_file_to_oss(file_path, file_type):
    """
    Generic upload function to OSS
    file_type: 'image', 'audio', 'video'
    """
    print(f"📤 Uploading local {file_type} to OSS: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        mime_map = {'image': 'image/png', 'audio': 'audio/wav', 'video': 'video/mp4'}
        mime_type = mime_map.get(file_type, 'application/octet-stream')
        
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, mime_type)}
            data = {'user_id': 'omni_test_user'}
            
            response = requests.post(
                f"{UPLOAD_BASE_URL}/api/{file_type}/upload",
                files=files,
                data=data,
                timeout=120 # Videos might take longer
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

def get_test_resource_url(resource_type):
    """
    Get URL for test resource (Image/Audio/Video).
    Priority: Local file upload -> Online URL
    """
    if resource_type == 'image':
        filename = LOCAL_IMAGE_NAME
        online_url = ONLINE_IMAGE_URL
    elif resource_type == 'audio':
        filename = LOCAL_AUDIO_NAME
        online_url = ONLINE_AUDIO_URL
    elif resource_type == 'video':
        filename = LOCAL_VIDEO_NAME
        online_url = ONLINE_VIDEO_URL
    else:
        return None

    local_path = os.path.join(TEST_FILES_DIR, filename)
    url_to_use = online_url
    
    if os.path.exists(local_path):
        print(f"📂 Found local {resource_type}: {local_path}")
        uploaded_url = upload_file_to_oss(local_path, resource_type)
        if uploaded_url:
            url_to_use = uploaded_url
        else:
            print(f"⚠️ Upload failed, falling back to online URL")
    else:
        print(f"⚠️ Local {resource_type} not found at {local_path}, using online URL")
        
    return url_to_use

# Don't define EXTENDED_MODELS content yet, we need to resolve URLs first at runtime
EXTENDED_MODELS_TEMPLATE = [
    {
        "id": "qwen3-omni-flash", # Omni supports Image + Audio
        "description": "Text + Image Input",
        "type": "image_input",
        "messages_template": [
            {
                "role": "user", 
                "content": [
                    {"image": "{IMAGE_URL}"},
                    {"text": "请描述这张图片。"}
                ]
            }
        ],
        "params": {
            "modalities": ["text", "audio"],
            "voice": "Ethan"
        }
    },
    {
        "id": "qwen3-omni-flash", # Now using Omni for Video
        "description": "Text + Video Input",
        "type": "video_input", 
        "messages_template": [
            {
                "role": "user",
                "content": [
                    {"video": "{VIDEO_URL}"}, # API Adapter handles 'video' key
                    {"text": "这段视频里发生了什么？"}
                ]
            }
        ],
        "params": {
            "modalities": ["text", "audio"], # Omni can output audio too
            "voice": "Ethan",
            "temperature": 0.1
        }
    },
    {
        "id": "qwen3-omni-flash",
        "description": "Text Input Only (No Audio)",
        "type": "text_only",
        "messages_template": [{"role": "user", "content": "你好，请简短介绍你自己。"}],
        "params": {
            "modalities": ["text"],
            "temperature": 0.7
        }
    },
    {
        "id": "qwen3-omni-flash",
        "description": "Text + Audio Input",
        "type": "audio_input", 
        "messages_template": [
            {
                "role": "user",
                "content": [
                    {"audio": "{AUDIO_URL}"},
                    {"text": "这段录音讲了什么？"}
                ]
            }
        ],
        "params": {
            "modalities": ["text", "audio"],
            "voice": "Cherry"
        }
    }
]

def test_omni_model(model_case):
    model_id = model_case["id"]
    desc = model_case["description"]
    params = model_case["params"]
    messages = model_case["messages"]
    
    print(f"Testing Model: {model_id} [{desc}] with params: {params}")
    
    payload = {
        "provider": "aliyun",
        "model": model_id,
        "messages": messages,
        "config": params
    }
    
    # Print input preview (truncate long URLs/Base64)
    preview = json.dumps(payload, ensure_ascii=False)
    if len(preview) > 500: preview = preview[:500] + "...}"
    print(f"\n[Input Data]: {preview}")
    
    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload, stream=True)
        if response.status_code == 200:
            print(f"✅ [PASS] {model_id} [{desc}] - Connection Established.")
            
            full_text_content = ""
            audio_data_buffer = b"" 
            error_detected = False
            audio_received = False
            
            os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            print(f"[Receiving Full Output]...")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if "Internal Error" in decoded_line or "Exception" in decoded_line:
                            print(f"❌ Error Line: {decoded_line}")
                            error_detected = True
                    
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            
                            # 1. Accumulate Text
                            current_text = ""
                            if "content" in data_json: 
                                current_text = data_json["content"]
                            elif "choices" in data_json and len(data_json["choices"]) > 0:
                                delta = data_json["choices"][0].get("delta", {})
                                if "content" in delta:
                                    current_text = delta["content"]
                                    
                                # 2. Accumulate Audio
                                if "audio" in delta and "data" in delta["audio"]:
                                    b64_data = delta["audio"]["data"]
                                    if b64_data:
                                        audio_data_buffer += base64.b64decode(b64_data)
                                        audio_received = True

                            if current_text:
                                full_text_content += current_text
                                
                        except Exception as e:
                            pass
            
            print(f"\n[Full Text Response Start] ---------------------------")
            print(full_text_content)
            print(f"[Full Text Response End] -----------------------------")
            
            # Save artifacts
            case_name = desc.replace(" ", "_").replace("->", "to").replace("+", "plus")[:30]
            
            # Save Text
            txt_filename = os.path.join(TEST_RESULTS_DIR, f"{model_id}_{case_name}_{timestamp}.txt")
            with open(txt_filename, "w", encoding="utf-8") as f:
                f.write(full_text_content)
            print(f"💾 Saved full text to: {os.path.abspath(txt_filename)}")
            
            # Save Audio
            if audio_received and audio_data_buffer:
                aud_filename = os.path.join(TEST_RESULTS_DIR, f"{model_id}_{case_name}_{timestamp}.wav")
                # Wav header logic...
                import wave
                try:
                    with wave.open(aud_filename, "wb") as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(24000)
                        wav_file.writeframes(audio_data_buffer)
                    print(f"💾 Saved full audio to: {os.path.abspath(aud_filename)}")
                except:
                    pass
            
            if error_detected:
                print(f"❌ [FAIL] {model_id} - Error detected during stream.")
                return False
                            
            print(f"   Stream finished. Content Length: {len(full_text_content)}, Audio Detected: {audio_received}")
            
            return True
        else:
            print(f"❌ [FAIL] {model_id} - Error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ [FAIL] {model_id} - Exception: {e}")
        return False

if __name__ == "__main__":
    import os
    for key in ["NO_PROXY", "no_proxy", "HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"]:
        if key in os.environ: del os.environ[key]

    print("--- Comprehensive Multimodal Model Test ---")
    print("1. Text + Image Input")
    print("2. Text + Video Input")
    print("3. Text + Audio Input")
    print("4. Text Only Input")
    print("-" * 60)
    
    # Pre-resolve URLs
    IMAGE_URL = get_test_resource_url('image')
    AUDIO_URL = get_test_resource_url('audio')
    VIDEO_URL = get_test_resource_url('video')
    
    print("-" * 60)
    
    results = {}
    for i, m_template in enumerate(EXTENDED_MODELS_TEMPLATE):
        model_case = m_template.copy()
        
        # Replace placeholders
        if "messages_template" in model_case:
            msg_str = json.dumps(model_case["messages_template"])
            msg_str = msg_str.replace("{IMAGE_URL}", IMAGE_URL or "")
            msg_str = msg_str.replace("{AUDIO_URL}", AUDIO_URL or "")
            msg_str = msg_str.replace("{VIDEO_URL}", VIDEO_URL or "")
            model_case["messages"] = json.loads(msg_str)
        
        print(f"\n--- Case {i+1}: {model_case['description']} ---")
        success = test_omni_model(model_case)
        results[f"Case_{i+1}_{model_case['description']}"] = "PASS" if success else "FAIL"
    
    print("\n--- Summary ---")
    for k, res in results.items():
        status_icon = "✅" if res == "PASS" else "❌"
        print(f"{status_icon} {k}: {res}")

