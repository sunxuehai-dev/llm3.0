import httpx
import json

def test_llama3_1_local():
    url = "http://localhost:8000/api/playground/chat"
    payload = {
        "provider": "local",
        "model": "llama3.1:8b",
        "messages": [
            {"role": "user", "content": "什么是世界大模型，有哪些模型列举一下"}
        ],
        "config": {
            "temperature": 0.7,
            "stream": False
        }
    }
    
    print(f"Connecting to {url}...")
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                print("\n--- Model Response ---")
                print(content)
                print("--- End of Response ---\n")
                print("Test Passed: Local Llama 3.1 8B is responding correctly.")
            else:
                print(f"Test Failed: Status code {response.status_code}")
                print(f"Response: {response.text}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_llama3_1_local()
