import requests
import json
import sys

# 后端地址
BASE_URL = "http://localhost:8000/api/playground/chat"

def test_frontend_simulation():
    print("--- 模拟前端请求: Qwen Omni 纯文本模式 ---")
    
    # 完全复刻您截图中的参数结构
    payload = {
        "provider": "aliyun",
        "model": "qwen3-omni-flash",
        "messages": [
            {
                "role": "user",
                "content": "你好"
            }
        ],
        "config": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 50,
            "repetition_penalty": 1.1,
            "enable_search": False,
            "enable_thinking": False,
            # 关键参数: 显式指定只返回文本
            "modalities": ["text"] 
        }
    }
    
    print(f"\n[发送请求载荷]: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    
    try:
        # 发送 POST 请求 (流式)
        response = requests.post(BASE_URL, json=payload, stream=True, timeout=60)
        
        if response.status_code == 200:
            print("\n✅ [连接成功] 后端响应 200 OK")
            print("[接收流式输出]...")
            
            full_content = ""
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    # 过滤 SSE 的 data: 前缀
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str.strip() == "[DONE]":
                            print("\n[传输结束]")
                            break
                        try:
                            data_json = json.loads(data_str)
                            # 解析 OpenAI 格式
                            if "choices" in data_json:
                                delta = data_json["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    print(content, end="", flush=True)
                                    full_content += content
                        except:
                            pass
            
            print(f"\n\n✅ 测试完成! 模型成功返回了纯文本内容。")
        else:
            print(f"\n❌ [请求失败] 状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"\n❌ [发生异常] 无法连接到后端: {e}")
        print("请检查: 1. start.bat 是否在运行? 2. 端口是否是 8000?")

if __name__ == "__main__":
    test_frontend_simulation()
