import requests
import json
import os
import datetime

# 获取项目根目录 (tests 的上一级)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")
os.makedirs(TEST_RESULTS_DIR, exist_ok=True)

BASE_URL = "http://localhost:8000/api/playground"

MODELS = [
    {
        "id": "doubao-1-5-pro-32k-250115",
        "description": "1.5 Pro (带参测试)",
        "params": {
            "temperature": 0.9,
            "top_p": 0.8,
            "max_tokens": 100,
            "frequency_penalty": 0.5
        }
    },
    {
        "id": "doubao-lite-32k-character-250228",
        "description": "Lite 角色扮演 (无参屏蔽测试)",
        "params": {
            "temperature": 1.2, # These shouldn't be passed down internally by the server
            "top_p": 0.9,
            "max_tokens": 200
        }
    },
    {
        "id": "doubao-1-5-lite-32k-250115",
        "description": "1.5 Lite (无参屏蔽测试)",
        "params": {
            "temperature": 0.5,
            "top_p": 0.5
        }
    }
]

def test_doubao_model(model_info):
    model_id = model_info["id"]
    params = model_info["params"]
    print(f"\n==============================================")
    print(f"Testing Model: {model_id} [{model_info['description']}]")
    print(f"Params sent from client: {params}")
    print(f"==============================================\n")
    
    payload = {
        "provider": "doubao",
        "model": model_id,
        "messages": [{"role": "user", "content": "什么是世界大模型。"}],
        "config": params
    }
    
    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload, stream=True)
        if response.status_code == 200:
            print(f"✅ [PASS] {model_id} - Connection Established.")
            
            full_content = ""
            error_detected = False
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            print(f"[Receiving Full Output]...")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if "Internal Error" in decoded_line or "Exception" in decoded_line or "Error" in decoded_line:
                        error_detected = True
                        print(f"❌ Error Line: {decoded_line}")
                    
                    if decoded_line.startswith("data: "):
                        data_str = decoded_line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data_json = json.loads(data_str)
                            content_piece = ""
                            
                            # Standard format
                            if "choices" in data_json and len(data_json["choices"]) > 0:
                                delta = data_json["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content_piece = delta["content"]
                            elif "content" in data_json:
                                content_piece = data_json["content"]
                                
                            if content_piece:
                                full_content += content_piece
                        except:
                            pass

            print(f"\n[Full Response Start] --------------------------------")
            print(full_content)
            print(f"[Full Response End] ----------------------------------\n")
            
            # Save to file
            filename = os.path.join(TEST_RESULTS_DIR, f"{model_id.replace('-', '_')}_output_{timestamp}.txt")
            with open(filename, "w", encoding="utf-8") as f:
                f.write(full_content)
            print(f"💾 Saved full response to: {os.path.abspath(filename)}")
            
            if error_detected or not full_content:
                print(f"❌ [FAIL] {model_id} - Error or empty content.")
                return False
            else:
                print(f"✅ [PASS] {model_id} - Stream completed successfully.")
                return True
        else:
            print(f"❌ [FAIL] {model_id} - Server HTTP Error {response.status_code}: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ [FAIL] Cannot connect to server at {BASE_URL}. Is the server running?")
        return False
    except Exception as e:
        print(f"❌ [FAIL] {model_id} - Exception: {e}")
        return False

if __name__ == "__main__":
    print("--- Doubao Text Model Routing Test ---")
    results = {}
    for m in MODELS:
        success = test_doubao_model(m)
        results[m["id"]] = "PASS" if success else "FAIL"
    
    print("\n--- Summary ---")
    for mid, res in results.items():
        print(f"{mid}: {res}")
