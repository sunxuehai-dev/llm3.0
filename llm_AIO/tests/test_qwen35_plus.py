import asyncio
import os
import sys
import json
from dotenv import load_dotenv

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

from app.adapters.aliyun import AliyunAdapter
from app.models.chat import ChatRequest, Message, ModelConfig

# Configuration
API_KEY = os.getenv("DASHSCOPE_API_KEY")
ENDPOINT_ID = "qwen3.5-plus"

# --- File Upload Utility for Multimodal Testing ---
import requests
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_FILES_DIR = os.path.join(PROJECT_ROOT, "test_files")
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")
UPLOAD_BASE_URL = "http://localhost:8000"
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

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
            data = {'user_id': 'qwen_test_user'}
            
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

async def run_chat(adapter, request, test_name):
    print(f"\n⏳ Submitting '{test_name}' Task...")
    reasoning = []
    content = []
    
    try:
        async for chunk in adapter.chat(request):
            if hasattr(chunk, 'reasoning_content') and chunk.reasoning_content:
                reasoning.append(chunk.reasoning_content)
                print(f"[Think]: {chunk.reasoning_content}", end="", flush=True)
            if chunk.content:
                # If we just switched from reasoning to content
                if reasoning and not content and chunk.content.strip():
                     print(f"\n[Response]: ", end="", flush=True)
                content.append(chunk.content)
                print(chunk.content, end="", flush=True)
            if chunk.finish_reason == "error":
                print(f"\n❌ Stream reported error: {chunk.content}")
        print("\n✅ Task Completed!")
        
        # Save logs
        safe_name = test_name.replace(" ", "_").lower()
        log_file = os.path.join(TEST_RESULTS_DIR, f"qwen35_{safe_name}_result.txt")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("=== REASONING ===\n")
            f.write("".join(reasoning) + "\n")
            f.write("=== CONTENT ===\n")
            f.write("".join(content) + "\n")
            
        print(f"💾 Saved log to {os.path.abspath(log_file)}")
        
    except Exception as e:
         print(f"❌ Exception in chat generator: {str(e)}")


async def test_qwen35_plus():
    print("=== Testing Qwen 3.5 Plus Adapter ===")
    
    if not API_KEY:
        print("Warning: DASHSCOPE_API_KEY environment variable not found.")
        return

    try:
        adapter = AliyunAdapter()
    except Exception as e:
        print(f"Failed to initialize adapter: {e}")
        return
        
    print(f"\n" + "="*50)
    print(f"🚀 Test 1: Complex Math Logic (Text Only)")
    print("="*50)
    request_text = ChatRequest(
        model=ENDPOINT_ID,
        provider="aliyun",
        messages=[
            Message(role="user", content="计算 1 到 100 中所有以3结尾或能被3整除的数字的总和。请先推理，后给出答案。")
        ],
        config=ModelConfig(enable_thinking=True, temperature=0.7)
    )
    await run_chat(adapter, request_text, "Math Logic text")

    
    print(f"\n" + "="*50)
    print(f"🚀 Test 2: Multimodal Image Understanding")
    print("="*50)
    image_url = upload_file_to_oss("1212.jpg", "image")
    if image_url:
        request_i2t = ChatRequest(
            model=ENDPOINT_ID,
            provider="aliyun",
            messages=[
                Message(role="user", content=[
                    {"type": "image", "image": image_url},
                    {"type": "text", "text": "这是什么图片？请仔细观察每一个细节并告诉我。"}
                ])
            ],
            config=ModelConfig(enable_thinking=True, temperature=0.7)
        )
        await run_chat(adapter, request_i2t, "Image Perception")
    else:
        print("Skipping Image test due to upload failure.")
        

    print(f"\n" + "="*50)
    print(f"🚀 Test 3: Multimodal Video Understanding")
    print("="*50)
    video_url = upload_file_to_oss("wan2.6-i2v_output_20260127_162717.mp4", "video")
    if video_url:
        request_v2t = ChatRequest(
            model=ENDPOINT_ID,
            provider="aliyun",
            messages=[
                Message(role="user", content=[
                    {"type": "video", "video": video_url},
                    {"type": "text", "text": "这段视频主要讲了什么事情？请一帧一帧地推理动作规律。"}
                ])
            ],
            config=ModelConfig(enable_thinking=True, temperature=0.7)
        )
        await run_chat(adapter, request_v2t, "Video Perception")
    else:
        print("Skipping Video test due to upload failure.")

    print(f"\n" + "="*50)
    print(f"🚀 Test 4: qwen3.5-flash Text Generation")
    print("="*50)
    request_flash = ChatRequest(
        model="qwen3.5-flash",
        provider="aliyun",
        messages=[
            Message(role="user", content="请解释一下量子纠缠。")
        ],
        config=ModelConfig(temperature=0.7, top_p=0.8, top_k=50, repetition_penalty=1.1, enable_search=False, enable_thinking=False)
    )
    await run_chat(adapter, request_flash, "qwen3.5-flash text")

    print(f"\n" + "="*50)
    print(f"🚀 Test 5: qwen3.5-35b-a3b Text Generation")
    print("="*50)
    request_35b = ChatRequest(
        model="qwen3.5-35b-a3b",
        provider="aliyun",
        messages=[
            Message(role="user", content="写一个关于未来城市的短诗。")
        ],
        config=ModelConfig(temperature=0.9, top_p=0.9, top_k=80, repetition_penalty=1.05, enable_search=False, enable_thinking=False)
    )
    await run_chat(adapter, request_35b, "qwen3.5-35b-a3b text")

if __name__ == "__main__":
    asyncio.run(test_qwen35_plus())
