import requests
import json
import sys
import os

# 获取项目根目录 (tests 的上一级)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_RESULTS_DIR = os.path.join(PROJECT_ROOT, "test_results")

BASE_URL = "http://localhost:8000/api/playground"
UPLOAD_BASE_URL = "http://localhost:8000"
TEST_FILES_DIR = os.path.join(PROJECT_ROOT, "test_files")

# --- Helper Functions for File Upload ---
def upload_file_to_server(file_path):
    """
    Upload local audio file to server to get a URL.
    Returns the URL if successful, None otherwise.
    """
    print(f"📤 Uploading local audio: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    try:
        # Determine mime type (basic)
        mime_type = 'audio/wav' if file_path.endswith('.wav') else 'audio/mpeg'
        
        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, mime_type)}
            # User ID is required by upload endpoint usually, using generic one
            data = {'user_id': 'asr_test_runner'}
            
            # Using the /video/upload or /audio/upload endpoint.
            # Based on file structure, likely /api/audio/upload or similar.
            # README says /api/{file_category}/upload. Let's use 'audio'.
            response = requests.post(
                f"{UPLOAD_BASE_URL}/api/audio/upload",
                files=files,
                data=data,
                timeout=60
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

def get_audio_url(file_name_or_url):
    """
    Resolve input to a valid URL.
    If it's already a URL, return it.
    If it's a local file name in test_files, upload it.
    If it's a local path elsewhere, try to upload it.
    """
    if file_name_or_url.startswith("http://") or file_name_or_url.startswith("https://"):
        return file_name_or_url
    
    # Check absolute path or relative to CWD
    if os.path.exists(file_name_or_url):
        return upload_file_to_server(file_name_or_url)
        
    # Check in TEST_FILES_DIR
    test_file_path = os.path.join(TEST_FILES_DIR, file_name_or_url)
    if os.path.exists(test_file_path):
        return upload_file_to_server(test_file_path)
    
    # Check relative to PROJECT_ROOT (legacy support)
    root_file_path = os.path.join(PROJECT_ROOT, file_name_or_url)
    if os.path.exists(root_file_path):
        return upload_file_to_server(root_file_path)

    print(f"⚠️ Could not find file: {file_name_or_url}")
    return None

# 语音识别模型配置
MODELS = [
    # Existing Realtime Model
    {
        "id": "qwen3-asr-flash",
        "description": "Realtime ASR (Basic)",
        "params": {
            "format": "wav",
            "sample_rate": 16000,
            "enable_punctuation_prediction": True,
            "enable_inverse_text_normalization": True,
            "disfluency_removal_enabled": False
        }
    },
    {
        "id": "qwen3-asr-flash",
        "description": "Realtime ASR (Disfluency Removal)",
        "params": {
            "format": "wav",
            "sample_rate": 16000,
            "enable_punctuation_prediction": True,
            "enable_inverse_text_normalization": False,
            "disfluency_removal_enabled": True
        }
    },
    # New File Trans Model (Auto-Semantic Break + Remove Disfluency)
    {
        "id": "qwen3-asr-flash-filetrans",
        "description": "File Trans (Semantic Break + No Disfluency)",
        "params": {
            "format": "wav",
            "sample_rate": 16000,
            "enable_punctuation_prediction": True,
            "enable_inverse_text_normalization": True,
            "disfluency_removal_enabled": True, # Test removing 'uh/ah'
            "enable_semantic_break": True       # Test semantic break
        }
    },
    # New FunASR (Speaker Diarization)
    {
        "id": "fun-asr",
        "description": "FunASR (Speaker Diarization)",
        "params": {
            "format": "wav",
            "sample_rate": 16000,
            "disfluency_removal_enabled": False,
            "speaker_diarization_enabled": True, # Test Speaker identification
            "max_speaker_count": 2
        }
    },

    # Xunfei Lfasr
    {
        "id": "xunfei-lfasr",
        "description": "Xunfei Lfasr (File Trans)",
        "params": {
            "format": "wav"
        }
    }
]

# 测试音频文件路径或URL
# 注意：需要准备实际的音频文件用于测试
# 可以是本地文件路径或网络URL
TEST_AUDIO_FILES = [
    # Use the file found in test_files directory by default
    "qwen3-omni-flash_Text_plus_Image_Input_20260127_163357.wav",
    # You can add external URLs here too
    # "https://dashscope.oss-cn-beijing.aliyuncs.com/samples/audio/paraformer/hello_world.wav"
]

def test_audio_transcription(model_info, audio_url, original_filename):
    """
    测试语音识别功能
    """
    model_id = model_info["id"]
    desc = model_info.get("description", model_id)
    params = model_info["params"]
    
    print(f"\n{'='*60}")
    print(f"测试模型: {model_id} - {desc}")
    print(f"原有文件名: {original_filename}")
    print(f"使用URL: {audio_url}")
    print(f"参数配置: {json.dumps(params, ensure_ascii=False, indent=2)}")
    print(f"{'='*60}")

    if not audio_url:
        print("❌ Skip: Invalid Audio URL")
        return False
        
    provider = "aliyun"
    if "xunfei" in model_id:
        provider = "xunfei"

    payload = {
        "provider": provider,
        "model": model_id,
        "input": audio_url, # Pass URL directly
        "config": params
    }
    
    print(f"\n[请求数据]: {json.dumps(payload, ensure_ascii=False, indent=2)}")
    
    try:
        response = requests.post(f"{BASE_URL}/audio/transcription", json=payload)
        
        print(f"\n[响应状态码]: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ [PASS] {model_id} - 请求成功")
            
            # 显示识别结果
            transcribed_text = result.get("text", "")
            raw_result = result.get("raw", "")
            
            print(f"\n[识别文本]:")
            print(f"{'-'*60}")
            print(transcribed_text)
            print(f"{'-'*60}")
            
            if raw_result:
                print(f"\n[原始响应]: {raw_result}")
            
            # 保存结果到文件
            import datetime
            os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                TEST_RESULTS_DIR, 
                f"asr_{model_id}_{os.path.basename(original_filename)}_{timestamp}.txt"
            )
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"模型: {model_id} ({desc})\n")
                f.write(f"源文件: {original_filename}\n")
                f.write(f"Audio URL: {audio_url}\n")
                f.write(f"参数配置:\n{json.dumps(params, ensure_ascii=False, indent=2)}\n")
                f.write(f"\n识别文本:\n{transcribed_text}\n")
                f.write(f"\n原始响应:\n{raw_result}\n")
            
            print(f"💾 结果已保存到: {os.path.abspath(filename)}")
            
            # 检查识别结果是否为空
            if not transcribed_text or len(transcribed_text.strip()) == 0:
                print(f"⚠️  [WARNING] 识别结果为空")
                return False
            
            return True
        else:
            error_detail = response.text
            print(f"❌ [FAIL] {model_id} - 错误 {response.status_code}: {error_detail}")
            
            # 保存错误信息
            import datetime
            os.makedirs(TEST_RESULTS_DIR, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(
                TEST_RESULTS_DIR,
                f"asr_{model_id}_error_{timestamp}.txt"
            )
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"模型: {model_id}\n")
                f.write(f"源文件: {original_filename}\n")
                f.write(f"错误状态码: {response.status_code}\n")
                f.write(f"错误详情: {error_detail}\n")
            
            return False
            
    except Exception as e:
        print(f"❌ [FAIL] {model_id} - 异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("语音识别（ASR）综合测试")
    print("="*60)
    
    if len(sys.argv) > 1:
        # 如果提供了命令行参数，使用指定的音频文件
        test_files = sys.argv[1:]
    else:
        # 否则使用默认的测试文件列表
        test_files = TEST_AUDIO_FILES
    
    if not test_files:
        print("\n⚠️  未指定测试音频文件")
        print("使用方法:")
        print("  python test_audio_transcription_comprehensive.py [音频文件路径或URL]")
        print("\n示例:")
        print("  python test_audio_transcription_comprehensive.py test_audio.mp3")
        print("  python test_audio_transcription_comprehensive.py https://example.com/audio.mp3")
        sys.exit(1)
    
    results = {}
    
    test_files_list = test_files
    
    results = {}
    
    # Process each file
    for audio_input in test_files_list:
        print(f"\nProcessing Audio Input: {audio_input}")
        
        # 1. Get URL (Upload if local)
        audio_url = get_audio_url(audio_input)
        
        if not audio_url:
            print(f"⏩ [SKIP] Generating URL failed for {audio_input}")
            continue

        # 2. Run tests for each model
        for model_info in MODELS:
            model_id = model_info["id"]
            desc = model_info.get("description", model_id)
            test_key = f"{model_id}_{os.path.basename(audio_input)}"
            
            success = test_audio_transcription(model_info, audio_url, audio_input)
            results[test_key] = "PASS" if success else "FAIL"
    
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for test_key, res in results.items():
        status_icon = "✅" if res == "PASS" else "❌"
        print(f"{status_icon} {test_key}: {res}")
    
    # 统计
    total = len(results)
    passed = sum(1 for r in results.values() if r == "PASS")
    failed = total - passed
    
    print(f"\n总计: {total} 个测试")
    print(f"通过: {passed} 个")
    print(f"失败: {failed} 个")
