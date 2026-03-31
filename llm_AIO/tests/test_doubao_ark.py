import asyncio
import os
import sys
from dotenv import load_dotenv

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

from app.adapters.doubao import DoubaoAdapter
from app.models.chat import ChatRequest, Message

# Configuration
# Please replace with your actual API Key (or set env var VOLCENGINE_API_KEY)
# Please replace with your actual Endpoint ID (e.g., ep-20240604111306-nb42j)
# If env var is not set, we will warn but continue trying for demo.
API_KEY = os.getenv("VOLCENGINE_API_KEY") or os.getenv("ARK_API_KEY")
MODELS_TO_TEST = [
    "doubao-seed-1-8-251228",
    "doubao-seed-2-0-pro-260215",
    "doubao-seed-2-0-mini-260215",
    "doubao-seed-1-6-251015",
    "doubao-seed-1-6-flash-250828"
]

# --- File Upload Utility for Multimodal Testing ---
import requests
import json

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

async def test_doubao_chat():
    print("=== Testing Doubao Adapter (Ark Runtime) ===")
    
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

    # Pre-upload files to avoid redundant uploads for each model
    print("\n--- Pre-uploading Test Files ---")
    image_url = upload_file_to_oss("1212.jpg", "image")
    video_url = upload_file_to_oss("wan2.6-i2v_output_20260127_162717.mp4", "video")

    for model_id in MODELS_TO_TEST:
        print(f"\n" + "="*50)
        print(f"🚀 Testing Model: {model_id}")
        print("="*50)

        # Test 1: Standard Text
        print(f"\n--- [Model: {model_id}] Test 1: Standard Text Input ---")
        request_text = ChatRequest(
            provider="doubao",
            model=model_id,  
            messages=[
                Message(role="user", content="给猫起三个名字")
            ],
            config={
                "temperature": 1,
                "max_tokens": 1024,
            }
        )
        await run_chat(adapter, request_text, f"{model_id}_text_input")

        # Test 2: Multimodal (Image)
        print(f"\n--- [Model: {model_id}] Test 2: Multimodal (Image Input) ---")
        if image_url:
            request_image = ChatRequest(
                provider="doubao",
                model=model_id,  
                messages=[
                    Message(role="user", content=[
                        {"type": "text", "text": "描述一下这张图片里的内容"},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ])
                ]
            )
            await run_chat(adapter, request_image, f"{model_id}_image_input")
        else:
            print("Skipping image test because file upload failed or server is not running.")
        
        # Test 3: Multimodal (Video)
        print(f"\n--- [Model: {model_id}] Test 3: Multimodal (Video Input) ---")
        if video_url:
            request_video = ChatRequest(
                provider="doubao",
                model=model_id,  
                messages=[
                    Message(role="user", content=[
                        {"type": "text", "text": "详细描述一下这段视频里发生了什么？"},
                        {"type": "video_url", "video_url": {"url": video_url}}
                    ])
                ],
                config={
                    "temperature": 0.7
                }
            )
            await run_chat(adapter, request_video, f"{model_id}_video_input")
        else:
            print("Skipping video test because file upload failed or server is not running.")
            
        # Test 4: Deep Thinking Toggle
        print(f"\n--- [Model: {model_id}] Test 4: Deep Thinking Toggle (enable_thinking=True) ---")
        request_thinking = ChatRequest(
            provider="doubao",
            model=model_id,  
            messages=[
                Message(role="user", content="9.11和9.8哪个数值更大？解释原因。")
            ],
            config={
                "temperature": 1 if "2-0-pro" in model_id else 0.7,
                "enable_thinking": True,
                "max_tokens": 1024
            }
        )
        await run_chat(adapter, request_thinking, f"{model_id}_thinking_enabled")

        # Test 5: Reasoning Effort (Only for models that support it)
        supports_reasoning = any(v in model_id for v in ["1-8", "2-0-mini", "2-0-pro", "2-0-lite"])
        if supports_reasoning:
            print(f"\n--- [Model: {model_id}] Test 5: Reasoning Effort (low vs high) ---")
            request_reasoning = ChatRequest(
                provider="doubao",
                model=model_id,  
                messages=[
                    Message(role="user", content="快速估算 321 乘以 123 的结果。")
                ],
                config={
                    "temperature": 1 if "2-0-pro" in model_id else 0.7,
                    "enable_thinking": True,
                    "reasoning_effort": "low",  # test low effort
                    "max_tokens": 1024
                }
            )
            await run_chat(adapter, request_reasoning, f"{model_id}_reasoning_low")
        else:
            print(f"\n--- [Model: {model_id}] Test 5 Skipped: Model does not support reasoning_effort ---")
            



async def run_chat(adapter, request, test_name=""):
    print(f"Sending request to model: {request.model}")
    print("Waiting for response...\n")
    
    import datetime
    full_content = ""
    os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

    try:
        async for chunk in adapter.chat(request):
            # Print reasoning content if available (e.g. thinking process)
            if hasattr(chunk, 'reasoning_content') and chunk.reasoning_content:
                print(f"[Thinking]: {chunk.reasoning_content}")
                full_content += f"[Thinking]: {chunk.reasoning_content}\n"
            
            if chunk.finish_reason == "stop":
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    full_content += chunk.content
                print("\n[Stream Finished]")
            elif chunk.finish_reason == "error":
                print(f"\n[Error]: {chunk.content}")
                full_content += f"\n[Error]: {chunk.content}"
            else:
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    full_content += chunk.content
                    
        # Save output to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = test_name.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "") if test_name else "Unknown"
        filename = os.path.join(TEST_RESULTS_DIR, f"doubao_{safe_name}_{timestamp}.txt")
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(full_content)
        print(f"\n💾 Saved result to: {os.path.abspath(filename)}")
        
    except Exception as e:
        print(f"\nAn error occurred during chat: {e}")

if __name__ == "__main__":
    asyncio.run(test_doubao_chat())
