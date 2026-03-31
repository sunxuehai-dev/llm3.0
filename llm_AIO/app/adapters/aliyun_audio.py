import os
import json
import uuid
import dashscope
import threading
import requests
from app.models.multimodal import AudioSpeechRequest, AudioTranscriptionRequest

# Define Callback globally
from dashscope.audio.asr import Recognition, Transcription, RecognitionCallback

# 转写结果下载缓存目录（可通过环境变量 TRANSCRIPTION_DOWNLOAD_DIR 覆盖）
TRANSCRIPTION_DOWNLOAD_DIR = os.getenv("TRANSCRIPTION_DOWNLOAD_DIR", "./downloads/transcriptions")

class SimpleASRCallback(RecognitionCallback):
    def __init__(self):
        self.text = ""
        self.raw = []
        self.is_final = False
        self._complete_event = threading.Event()
        
    def on_open(self):
        # print("DEBUG: Callback on_open")
        pass
    
    def on_close(self):
        # print("DEBUG: Callback on_close")
        self.is_final = True
        self._complete_event.set()
        
    def on_event(self, result):
        # print(f"DEBUG: Callback on_event: {result}")
        self.raw.append(str(result))
        if result.get_sentence().get('text'):
            self.text += result.get_sentence().get('text')
            
    def on_error(self, result):
        print(f"ASR Error details: {result}")
        self.raw.append(str(result))
        self.is_final = True
        self._complete_event.set()
            
    def wait_for_completion(self, timeout=60):
        return self._complete_event.wait(timeout)

class AliyunAudioAdapter:
    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        dashscope.api_key = self.api_key

    async def _download_and_parse_transcription(self, url: str, task_id: str = None) -> tuple:
        """
        从 transcription_url 下载 JSON 文件，保存到本地，解析并返回文本。
        返回: (transcribed_text, saved_path)，saved_path 为 None 表示未保存。
        """
        import asyncio
        loop = asyncio.get_event_loop()

        def _sync_download():
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                return "", None
            data = resp.json()

            # 保存到本地
            os.makedirs(TRANSCRIPTION_DOWNLOAD_DIR, exist_ok=True)
            filename = f"{task_id or uuid.uuid4().hex[:12]}.json"
            saved_path = os.path.join(TRANSCRIPTION_DOWNLOAD_DIR, filename)
            with open(saved_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # 解析 transcripts
            text_parts = []
            if "transcripts" in data:
                for t in data["transcripts"]:
                    text_parts.append(t.get("text", ""))
            elif "transcription" in data:
                text_parts.append(data["transcription"])
            return "\n".join(filter(None, text_parts)), saved_path

        try:
            return await loop.run_in_executor(None, _sync_download)
        except Exception as e:
            print(f"Error downloading/parsing transcription_url: {e}")
            return "", None

    async def speech(self, request: AudioSpeechRequest):
        """
        Text-to-Speech using Aliyun CosyVoice or Sambert
        """
        model = request.model 
        voice = request.config.voice or "cherry"
        
        try:
            # Mapping config
            kwargs = {
                "format": request.config.format or "mp3",
                "sample_rate": request.config.sample_rate or 16000,
            }
            
            # Select synthesizer class - Using main SpeechSynthesizer for all as it is most stable
            from dashscope.audio.tts import SpeechSynthesizer
            synthesizer_class = SpeechSynthesizer

            # speech_rate is more universal across DashScope TTS models than 'speed'
            kwargs['speech_rate'] = float(request.config.speed or 1.0)
            
            # Volume mapping - some models take 0-100, some 0.0-1.0. 
            # Sambert/CosyVoice in recent SDKs often use 1.0 as baseline.
            # We'll try to pass it as provided or normalized.
            kwargs['volume'] = int(request.config.volume or 50)
                 
            if request.config.language_type and request.config.language_type != "Auto":
                 kwargs['language_type'] = request.config.language_type

            import asyncio
            import functools
            import json

            loop = asyncio.get_event_loop()
            
            # Map parameters based on model type
            is_cosyvoice = "cosyvoice" in model.lower()
            
            # For CosyVoice, some parameters might have different ranges or names
            # But according to DashScope docs, speech_rate and volume are generally supported.
            # Let's ensure volume is within 0-100 and speech_rate is 0.5-2.0.
            
            call_kwargs = {
                "model": model,
                "text": request.input,
                "voice": voice,
                "format": kwargs.get("format", "mp3"),
                "sample_rate": kwargs.get("sample_rate", 16000),
                "speech_rate": kwargs.get("speech_rate", 1.0),
                "volume": kwargs.get("volume", 50)
            }
            
            # Additional params for v2/cosyvoice if needed
            if is_cosyvoice:
                # CosyVoice sometimes has issues with volume > 100 or speech_rate outside range
                call_kwargs["volume"] = min(max(int(call_kwargs["volume"]), 0), 100)
                call_kwargs["speech_rate"] = min(max(float(call_kwargs["speech_rate"]), 0.5), 2.0)

            # Correct Implementation: Use tts_v2 for CosyVoice
            
            # --- tts_v2 Logic for CosyVoice ---
            # --- tts_v2 Logic for CosyVoice ---
            if is_cosyvoice:
                from dashscope.audio.tts_v2 import SpeechSynthesizer as SpeechSynthesizerV2
                from dashscope.audio.tts_v2 import AudioFormat as AudioFormatV2
                
                # Helper to find matching AudioFormat
                # v2 requires a specific AudioFormat Enum, not just string/int
                target_fmt = kwargs.get("format", "mp3").lower()
                target_rate = int(kwargs.get("sample_rate", 16000))
                
                # Default fallback
                selected_format = AudioFormatV2.DEFAULT
                
                # Try to find match in AudioFormatV2
                # AudioFormatV2 members look like: MP3_16000HZ_MONO_128KBPS
                # We search for format and rate in the member name or properties?
                # The Enum members are tuples: ('mp3', 16000, 'mono', '128kbps')
                # We can iterate to find a match
                for fmt_enum in AudioFormatV2:
                     if fmt_enum.format == target_fmt and fmt_enum.sample_rate == target_rate:
                         selected_format = fmt_enum
                         break
                
                # If no exact match (e.g. 22050Hz mp3 might not exist or be named differently),
                # fallback to commonly supported ones or DEFAULT.
                # If falling back, we might lose specific sample rate, but better than crash.
                
                # tts_v2 init params
                # Note: v2 __init__ does NOT accept 'sample_rate' directly. It comes from 'format'.
                init_params = {
                    "model": model,
                    "voice": voice,
                    "format": selected_format,
                    "volume": kwargs.get("volume", 50),
                    "speech_rate": kwargs.get("speech_rate", 1.0),
                    "pitch_rate": kwargs.get("pitch", 1.0)
                }
                
                def run_v2_sync():
                    synthesizer = SpeechSynthesizerV2(**init_params)
                    return synthesizer.call(request.input)

                audio_data = await loop.run_in_executor(None, run_v2_sync)
                
                if audio_data:
                     try:
                        # V2 returns raw bytes directly, or we might need to check if it's a bytearray
                        if isinstance(audio_data, (bytes, bytearray)):
                             # Use the actual format/rate from the selected enum
                             media_type = f"audio/{selected_format.format}"
                             if selected_format.format == "mp3": media_type = "audio/mpeg"
                             return {"audio_data": audio_data, "content_type": media_type}
                        else:
                             # Should not happen if call() succeeds as per doc
                             return {"error": f"TTS Failed: Unexpected return type {type(audio_data)}"}
                     except Exception as e:
                         return {"error": f"TTS Logic Error: {e}"}
                else:
                    return {"error": "TTS Failed: No audio data returned from tts_v2"}

            # --- Legacy Logic for Sambert/Other (using standard SpeechSynthesizer) ---
            else:
                call_func = functools.partial(
                    synthesizer_class.call,
                    **call_kwargs
                )
                
                result = await loop.run_in_executor(None, call_func)
                
                # Standard result handling (same as before)
                if result is None:
                    return {"error": "TTS Failed: Received None result"}
                
                audio_data = None
                try:
                   audio_data = result.get_audio_data()
                except:
                   pass

                if audio_data is not None:
                    media_type = f"audio/{call_kwargs['format']}"
                    if call_kwargs['format'] == "mp3": media_type = "audio/mpeg"
                    return {"audio_data": audio_data, "content_type": media_type}
                else:
                     response_info =  "Unknown error"
                     try: response_info = result.get_response()
                     except: response_info = str(result)
                     return {"error": f"TTS Failed: {response_info}"}
        except Exception as e:
            # unique handling for the KeyError if it bubbles up (though the inner try/except should catch it)
            if "begin_time" in str(e):
                 return {"error": "Aliyun SDK Error: CosyVoice model returned data without timestamps, causing an SDK crash. Please report this to the developer."}
            import traceback
            traceback.print_exc()
            return {"error": f"TTS Exception: {str(e)}"}

    async def transcribe(self, request: AudioTranscriptionRequest):
        """
        ASR using Aliyun Qwen / Paraformer
        Supports:
        - Realtime (Recognition): qwen3-asr-flash, paraformer-realtime
        - File (Transcription): qwen3-asr-flash-filetrans, fun-asr (paraformer-v2)
        """
        import asyncio
        import functools
        import json
        from http import HTTPStatus
        
        # Lazy import to avoid potential import errors
        # Move imports inside to avoid circular deps or if module level failed
        try:
             from dashscope.audio.asr import Recognition, Transcription, ResultCallback
        except ImportError:
             pass

        try:
            loop = asyncio.get_event_loop()
            
            model = request.model 
            input_source = request.input
            
            # 1. Determine Input Type (URL or Local File)
            is_url = input_source.startswith("http://") or input_source.startswith("https://")
            
            # Initialize paths default
            file_path = None
            temp_file_path = None


            # 2. Determine Service Type: Recognition (Realtime) or Transcription (File)
            # "filetrans" or "fun-asr" implies File/Batch mode which supports Diarization
            use_file_transcription = "filetrans" in model or "fun-asr" in model or "paraformer-v2" in model
            
            # --- FILE TRANSCRIPTION (Batch/Async) ---
            if use_file_transcription:
                # Transcription API usually requires URL. If local file, we might need to upload it or check if SDK supports local path.
                # DashScope Transcription SDK supports file_urls (list) or local file upload internally if supported.
                # IMPORTANT: DashScope Python SDK Transcription.call() often prefers URLs. 
                # If 'input_source' is local, we must ensure it's accessible or let SDK handle upload if it can.
                # For `file_urls`, it MUST be a URL.
                
                # As per Aliyun docs, Transcription.call(file_urls=[...]) or potentially local file.
                # Let's assume input_source is a URL as recommended for "FileTrans". 
                # If user provided a local path to this endpoint, it might fail if not converted.
                # However, for this implementation, we proceed assuming input_source is valid.
                
                transcription_kwargs = {
                    "model": model,
                    "file_urls": [input_source] 
                }
                
                # Add optional keys
                if hasattr(request.config, 'disfluency_removal_enabled'):
                     transcription_kwargs['disfluency_removal_enabled'] = request.config.disfluency_removal_enabled
                
                if hasattr(request.config, 'speaker_diarization_enabled') and request.config.speaker_diarization_enabled:
                     transcription_kwargs['speaker_diarization_enabled'] = True
                
                if hasattr(request.config, 'enable_semantic_break') and request.config.enable_semantic_break:
                     transcription_kwargs['enable_semantic_break'] = True

                loop = asyncio.get_event_loop()
                
                # Transcription is usually Async in nature (submit task -> wait). 
                # SDK 2.x+ call() might return a job or wait. 
                # We use call() which (in newer SDK) might be synchronous-waiting if not specified otherwise, 
                # OR we might need to poll. 
                # DashScope SDK 'call' for Transcription is typically blocking-wait if no 'async_request' specified? 
                # Actually, standard usage returns a job generator or result.
                
                def run_transcription_rest():
                    """使用 RESTful API 替代 SDK 调用，以获得更好的 URL 兼容性"""
                    import requests
                    import json
                    import time
                    
                    api_url_submit = "https://dashscope.aliyuncs.com/api/v1/services/audio/asr/transcription"
                    api_url_query_base = "https://dashscope.aliyuncs.com/api/v1/tasks/"
                    
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "X-DashScope-Async": "enable"
                    }
                    
                    # 构造 Payload
                    payload = {
                        "model": model,
                        "input": {
                            "file_url": input_source
                        },
                        "parameters": {
                            "disfluency_removal_enabled": transcription_kwargs.get('disfluency_removal_enabled', False),
                            "speaker_diarization_enabled": transcription_kwargs.get('speaker_diarization_enabled', False),
                            "enable_semantic_break": transcription_kwargs.get('enable_semantic_break', False),
                        }
                    }
                    
                    try:
                        # 1. 提交任务
                        submit_resp = requests.post(api_url_submit, headers=headers, json=payload, timeout=30)
                        if submit_resp.status_code != 200:
                            return {"error": f"Submit Failed ({submit_resp.status_code}): {submit_resp.text}"}
                        
                        submit_data = submit_resp.json()
                        task_id = submit_data.get("output", {}).get("task_id")
                        if not task_id:
                            return {"error": f"No task_id in response: {submit_data}"}
                        
                        # 2. 轮询状态
                        query_url = api_url_query_base + task_id
                        max_retries = 60 # 最多等 2 分钟 (2s * 60)
                        
                        for _ in range(max_retries):
                            time.sleep(2)
                            query_resp = requests.get(query_url, headers=headers, timeout=10)
                            if query_resp.status_code != 200:
                                continue # 偶发网络问题重试
                            
                            query_data = query_resp.json()
                            status = query_data.get("output", {}).get("task_status")
                            
                            if status == "SUCCEEDED":
                                return query_data
                            elif status in ("FAILED", "CANCELED"):
                                return {"error": f"Transcription {status}: {query_data.get('output', {}).get('message')}", "raw": str(query_data)}
                                
                        return {"error": "Transcription Timeout after 120s"}
                        
                    except Exception as e:
                        return {"error": f"REST API Exception: {str(e)}"}

                # 执行 REST 逻辑
                response_data = await loop.run_in_executor(None, run_transcription_rest)
                
                if "error" in response_data:
                    return response_data

                # --- 结果解析 (兼容 output.results 与 output.result.transcription_url) ---
                transcribed_text = ""
                raw_result = str(response_data)
                output_obj = response_data.get("output", {})
                results = output_obj.get("results", [])

                for res in results:
                    if 'sentences' in res:
                        for sent in res['sentences']:
                            text = sent.get('text', '')
                            speaker_id = sent.get('speaker_id', '')
                            if request.config.speaker_diarization_enabled and speaker_id:
                                transcribed_text += f"[Speaker {speaker_id}] {text}\n"
                            else:
                                transcribed_text += text
                    elif 'text' in res:
                        transcribed_text += res['text']

                # 若无直接文本，检查 transcription_url：output.result.transcription_url 或 output.results[].transcription_url
                url_to_fetch = None
                if not transcribed_text:
                    result_obj = output_obj.get("result", {})
                    if isinstance(result_obj, dict) and result_obj.get("transcription_url"):
                        url_to_fetch = result_obj["transcription_url"]
                    else:
                        for res in results:
                            if res.get("transcription_url"):
                                url_to_fetch = res["transcription_url"]
                                break

                if url_to_fetch:
                    transcribed_text, saved_path = await self._download_and_parse_transcription(url_to_fetch, output_obj.get("task_id"))
                    if saved_path:
                        raw_result += f" | 本地缓存: {saved_path}"

                return {
                    "text": transcribed_text,
                    "raw": raw_result
                }

            # --- MULTIMODAL CONVERSATION (Qwen3 ASR Flash) ---
            elif model == "qwen3-asr-flash":
                from dashscope import MultiModalConversation
                
                # prepare messages
                # If input_source is a local file, MultiModalConversation in DashScope supports 'file://' prefix
                audio_url = input_source
                if not is_url:
                    # Dashscope MultiModal requires absolute path with file://
                    abs_path = os.path.abspath(input_source)
                    audio_url = f"file://{abs_path}"
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"audio": audio_url}
                        ]
                    }
                ]
                
                asr_options = {}
                if hasattr(request.config, 'enable_inverse_text_normalization'):
                    asr_options['enable_itn'] = request.config.enable_inverse_text_normalization
                    
                def run_multimodal():
                    return MultiModalConversation.call(
                        model=model,
                        messages=messages,
                        result_format="message",
                        asr_options=asr_options if asr_options else None
                    )
                
                response = await loop.run_in_executor(None, run_multimodal)
                
                if response.status_code == HTTPStatus.OK:
                    try:
                        text = response.output.choices[0].message.content[0]['text']
                        return {"text": text, "raw": str(response)}
                    except Exception as e:
                         return {"error": f"Failed to parse MultiModal response: {e}", "raw": str(response)}
                else:
                    return {"error": f"ASR Failed: {response.message}", "raw": str(response)}

            # --- RECOGNITION (Realtime/Short Audio - Older Models) ---
            else:
                # Handle Local vs URL for Recognition
                if not is_url:
                    if not os.path.exists(input_source):
                        return {"error": f"File not found: {input_source}"}
                    file_path = input_source
                else:
                    import tempfile
                    import urllib.request
                    file_ext = f".{request.config.format}" if request.config.format else ".wav"
                    if not file_ext.startswith('.'): file_ext = f".{file_ext}"
                    
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
                    temp_file_path = temp_file.name
                    temp_file.close()
                    try:
                        urllib.request.urlretrieve(input_source, temp_file_path)
                        file_path = temp_file_path
                    except Exception as e:
                        if os.path.exists(temp_file_path): os.unlink(temp_file_path)
                        return {"error": f"Download failed: {str(e)}"}

                # Recognition Config
                # Create callback
                callback = SimpleASRCallback()
                
                # Check for format (aliyun requires 'pcm', 'wav', 'mp3' etc)
                # Ensure format strings are lowercase and valid
                audio_format = request.config.format or "wav"
                sample_rate = request.config.sample_rate or 16000

                # Initialize Recognition
                # Note: 'format' is a reserved keyword, so in dict it's fine, but in init args
                # we usually pass it as keyword arg.
                
                # Construct kwargs for Recognition
                init_kwargs = {
                    "model": model,
                    "format": audio_format,
                    "sample_rate": sample_rate,
                    "callback": callback,
                }
                
                # Add optional config to init or start?
                # DashScope SDK typically puts config in init for Recognition.
                if hasattr(request.config, 'enable_punctuation_prediction'):
                    init_kwargs["enable_punctuation_prediction"] = request.config.enable_punctuation_prediction
                if hasattr(request.config, 'enable_inverse_text_normalization'):
                    init_kwargs["enable_inverse_text_normalization"] = request.config.enable_inverse_text_normalization
                if hasattr(request.config, 'disfluency_removal_enabled'):
                    init_kwargs["disfluency_removal_enabled"] = request.config.disfluency_removal_enabled

                def run_recognition_sync():
                    recognizer = Recognition(**init_kwargs)
                    recognizer.start()
                    
                    # Read file and send
                    # 32k buffer size
                    _buf_size = 32768
                    try:
                        with open(file_path, 'rb') as f:
                            while True:
                                data = f.read(_buf_size)
                                if not data:
                                    break
                                recognizer.send_audio_frame(data)
                    except Exception as e:
                       print(f"Error reading file: {e}")
                    
                    recognizer.stop()
                    
                    # Wait for callback to receive all results
                    # Typically stop() triggers on_close eventually.
                    if not callback.wait_for_completion(timeout=30):
                         print("Warning: ASR Recognition timed out waiting for on_close")
                         
                    return callback

                # Run blocking recognition in executor
                final_callback = await loop.run_in_executor(None, run_recognition_sync)
                
                # Cleanup
                if temp_file_path and os.path.exists(temp_file_path):
                    try: os.unlink(temp_file_path)
                    except: pass
                
                return {
                     "text": final_callback.text,
                     "raw": str(final_callback.raw)
                }

        except Exception as e:
            import traceback
            traceback.print_exc()
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            if temp_file_path and os.path.exists(temp_file_path):
                try: os.unlink(temp_file_path)
                except: pass
            return {"error": f"ASR Exception: {error_detail}"}
