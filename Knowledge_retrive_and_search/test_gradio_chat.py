import gradio as gr

# 测试 Gradio 6.0+ Chatbot 的消息格式和 State 更新

def test_chat(message, history):
    """测试聊天函数"""
    print(f"Received history type: {type(history)}, value: {history}")
    
    # 添加用户消息和临时回复
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": "正在生成..."}]
    print(f"After adding messages: {history}")
    
    # 模拟流式输出 - 更新最后一条消息
    import time
    for i in range(3):
        time.sleep(0.5)
        # 更新助手回复
        history[-1]["content"] = f"正在生成... {i+1}/3"
        print(f"Yielding: {history}")
        yield history
    
    # 最终回复
    history[-1]["content"] = f"你说的是: {message}"
    yield history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="测试聊天")
    msg = gr.Textbox(label="输入消息")
    
    # 注意：在 Gradio 6.0+ 中，State 作为输入和输出参数
    state = gr.State([])
    
    msg.submit(test_chat, [msg, state], [chatbot])

demo.launch(server_port=7862)
