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
        "id": "qwen-max",
        "params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "enable_search": True
            # enable_thinking NOT supported
        }
    },
    {
        "id": "qwen-plus",
        "params": {
            "temperature": 0.5,
            "top_p": 0.9,
            "top_k": 20,
            "repetition_penalty": 1.0,
            "enable_search": True,
            "enable_thinking": False
        }
    },
    {
        "id": "qwen-turbo",
        "params": {
            "temperature": 1.5,
            "top_p": 0.5,
            "top_k": 80,
            "repetition_penalty": 1.2,
            "enable_search": False,
            "enable_thinking": True
        }
    },
    {
        "id": "deepseek-v3.2",
        "params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 50,
            "repetition_penalty": 1.05,
            "enable_search": False,
            "enable_thinking": True
        }
    }
]

def test_text_model(model_info):
    model_id = model_info["id"]
    params = model_info["params"]
    print(f"Testing Model: {model_info['id']} with params: {params}")
    
    payload = {
        "provider": "aliyun",
        "model": model_id,
        "messages": [{"role": "user", "content": "你知道世界大模型吗。"}],
        "config": params
    }
    print(f"\n[Input Data]: {json.dumps(payload, ensure_ascii=False)}")
    
    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload, stream=True)
        if response.status_code == 200:
            print(f"✅ [PASS] {model_id} - Connection Established.")
            
            # Prepare for full content capture
            full_content = ""
            error_detected = False
            
            # Create output directory if not exists
            import datetime
            os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            print(f"[Receiving Full Output]...")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if "Internal Error" in decoded_line or "Exception" in decoded_line:
                        error_detected = True
                        print(f"❌ Error Line: {decoded_line}")
                    
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
            
            # Save to file
            filename = os.path.join(TEST_RESULTS_DIR, f"{model_id}_output_{timestamp}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(full_content)
            print(f"💾 Saved full response to: {os.path.abspath(filename)}")
            
            if error_detected:
                print(f"❌ [FAIL] {model_id} - Error detected in response content.")
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
    print("--- Comprehensive Text Model Test ---")
    results = {}
    for m in MODELS:
        success = test_text_model(m)
        results[m["id"]] = "PASS" if success else "FAIL"
    
    print("\n--- Summary ---")
    for mid, res in results.items():
        print(f"{mid}: {res}")
