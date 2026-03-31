from typing import List, Dict, Any, Union

class ModelRegistry:
    """
    Central repository for model metadata.
    Defines available models across Text, Audio, and Visual modalities.
    """
    
    # --- Common Parameters ---
    _TEMP_PARAM = {
        "name": "temperature", "label": "随机性 (Temperature)", "type": "slider",
        "min": 0.0, "max": 2.0, "default": 0.7, "step": 0.1,
        "description": "值越低越严谨，值越高越发散创新。"
    }
    
    _TOP_P_PARAM = {
        "name": "top_p", "label": "核采样 (Top-P)", "type": "slider",
        "min": 0.0, "max": 1.0, "default": 0.8, "step": 0.05,
        "description": "仅从概率最高的 Top-P Token 中进行采样，缩小选择范围。"
    }

    _TOP_K_PARAM = {
        "name": "top_k", "label": "候选数 (Top-K)", "type": "slider",
        "min": 1, "max": 100, "default": 50, "step": 1,
        "description": "仅保留前 K 个最可能的 Token。"
    }

    _REP_PENALTY_PARAM = {
        "name": "repetition_penalty", "label": "重复惩罚 (Penalty)", "type": "slider",
        "min": 1.0, "max": 2.0, "default": 1.1, "step": 0.05,
        "description": "值越大，越为了不重复而强行换词。"
    }

    _SEARCH_PARAM = {
        "name": "enable_search", "label": "联网思考 (Search)", "type": "switch",
        "default": False,
        "description": "开启后，模型会先上网搜索再回答。"
    }

    _THINKING_PARAM = {
        "name": "enable_thinking", "label": "深度思考 (Think)", "type": "switch",
        "default": False,
        "description": "开启混合思考模式 (CoT)，让模型进行深度推理。"
    }

    # --- Local Model Specific Parameters ---
    _MAX_TOKENS_PARAM = {
        "name": "max_tokens", "label": "最大长度 (Max Tokens)", "type": "slider",
        "min": 128, "max": 4096, "default": 2048, "step": 128,
        "description": "生成回复的最大 Token 数量。"
    }

    _SEED_PARAM = {
        "name": "seed", "label": "随机种子 (Seed)", "type": "number",
        "default": None,
        "description": "固定种子可使生成结果可复现。留空为随机。"
    }

    _PRESENCE_PENALTY_PARAM = {
        "name": "presence_penalty", "label": "话题惩罚 (Presence)", "type": "slider",
        "min": -2.0, "max": 2.0, "default": 0, "step": 0.1,
        "description": "正值鼓励新话题，负值倾向已有话题。"
    }
    
    _REASONING_EFFORT_PARAM = {
        "name": "reasoning_effort", "label": "推理强度", "type": "select",
        "options": ["minimal", "low", "medium", "high"],
        "default": "medium",
        "description": "控制模型深度思考的工作量。minimal(不思考)、low(轻量)、medium(均衡)、high(深度分析)。"
    }

    _FIXED_TEMP_PARAM = {
        "name": "temperature", "label": "随机性 (固定)", "type": "slider",
        "min": 1.0, "max": 1.0, "default": 1.0, "step": 0.1,
        "description": "该模型强制固定值为 1.0，忽略调整以保障稳定性。"
    }

    _FIXED_TOP_P_PARAM = {
        "name": "top_p", "label": "核采样 (固定)", "type": "slider",
        "min": 0.95, "max": 0.95, "default": 0.95, "step": 0.05,
        "description": "该模型强制固定值为 0.95，忽略调整以保障稳定性。"
    }

    _FREQUENCY_PENALTY_PARAM = {
        "name": "frequency_penalty", "label": "频率惩罚 (Frequency)", "type": "slider",
        "min": -2.0, "max": 2.0, "default": 0, "step": 0.1,
        "description": "正值降低高频词，负值倾向常用词。"
    }

    _IMAGE_SIZE_PARAM = {
        "name": "size", "label": "图片尺寸", "type": "select",
        "options": ["1024x1024", "720x1280", "1280x720", "1664x928", "1472x1104", "1328x1328", "1104x1472", "928x1664",
                    "2K", "3K", "4K"], # Added general resolution options
        "default": "1024x1024" # Note: qwen-image-max default is 1664x928 but we keep loose default or handle in UI
    }

    _NEGATIVE_PROMPT_PARAM = {
        "name": "negative_prompt", "label": "反向提示词", "type": "text",
        "description": "不希望出现的内容，支持中英文，上限500字符。"
    }

    _PROMPT_EXTEND_PARAM = {
        "name": "prompt_extend", "label": "智能改写", "type": "switch",
        "default": True,
        "description": "是否开启 Prompt 智能改写优化。"
    }

    _WATERMARK_PARAM = {
        "name": "watermark", "label": "添加水印", "type": "switch",
        "default": True,
        "description": "在生成的图片右下角添加AI生成字样"
    }

    _SEQ_IMG_GEN_PARAM = {
        "name": "sequential_image_generation", "label": "组图模式", "type": "select",
        "options": ["auto", "disabled"], "default": "disabled",
        "description": "开启后，若提示词需要，模型会一次性连贯生成最多15张关联的参考图片组。"
    }
    
    # --- Video Generation Specific Parameters ---
    _VIDEO_RESOLUTION_PARAM = {
        "name": "resolution", "label": "视频分辨率", "type": "select",
        "options": ["480p", "720p", "1080p"],
        "default": "720p"
    }

    _VIDEO_RATIO_PARAM = {
        "name": "ratio", "label": "画面比例", "type": "select",
        "options": ["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"],
        "default": "16:9",
        "description": "Adaptive 将自动根据上传的图片调整比例。"
    }

    _VIDEO_DURATION_PARAM = {
        "name": "duration", "label": "生成时长 (秒)", "type": "slider",
        "min": 2, "max": 12, "default": 5, "step": 1,
        "description": "指定生成的视频长度（秒）。"
    }

    _VIDEO_CAMERA_FIXED_PARAM = {
        "name": "camera_fixed", "label": "固定镜头", "type": "switch",
        "default": False,
        "description": "平台会在提示词中追加固定摄像头的指令。图生视频-参考图模式不支持。"
    }

    _VIDEO_WATERMARK_PARAM = {
        "name": "watermark", "label": "添加水印", "type": "switch",
        "default": False,
        "description": "在生成的视频中包含水印。"
    }
    
    # --- Doubao Video specific ---
    _DOUBAO_VIDEO_RESOLUTION_PARAM = {
        "name": "resolution", "label": "视频分辨率", "type": "select",
        "options": ["480p", "720p", "1080p"],
        "default": "720p"
    }

    _DOUBAO_VIDEO_DURATION_PARAM = {
        "name": "duration", "label": "生成时长 (秒)", "type": "slider",
        "min": 2, "max": 12, "default": 5, "step": 1,
        "description": "指定生成的视频长度（秒）。"
    }
    
    _DOUBAO_VIDEO_RATIO_PARAM = {
        "name": "ratio", "label": "画面比例", "type": "select",
        "options": ["16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive"],
        "default": "16:9",
        "description": "Adaptive 将自动根据上传的首帧图片调整比例"
    }
    
    _VOICE_SPEED_PARAM = {
        "name": "speed", "label": "语速 (Speed)", "type": "slider",
        "min": 0.5, "max": 2.0, "default": 1.0, "step": 0.1
    }

    _VOLUME_PARAM = {
        "name": "volume", "label": "音量 (Volume)", "type": "slider",
        "min": 0, "max": 100, "default": 50, "step": 5
    }

    # --- TTS Voice Lists (Updated from Excel) ---
    _COSYVOICE_V3_FLASH_OPTIONS = [
        "longanyang (阳光大男孩)",
        "longanhuan (欢脱元气女)",
        "longhuhu_v3 (天真烂漫女童)",
        "longpaopao_v3 (飞天泡泡音)",
        "longjielidou_v3 (阳光顽皮男)",
        "longxian_v3 (豪放可爱女)",
        "longling_v3 (稚气呆板女)",
        "longshanshan_v3 (戏剧化童声)",
        "longniuniu_v3 (阳光男童声)",
        "longantai_v3 (嗲甜台湾女)",
        "longhua_v3 (元气甜美女)",
        "longcheng_v3 (智慧青年男)",
        "longze_v3 (温暖元气男)",
        "longzhe_v3 (呆板大暖男)",
        "longyan_v3 (温暖春风女)",
        "longxing_v3 (温婉邻家女)",
        "longtian_v3 (磁性理智男)",
        "longwan_v3 (细腻柔声女)",
        "longqiang_v3 (浪漫风情女)",
        "longfeifei_v3 (甜美娇气女)",
        "longhao_v3 (多情忧郁男)",
        "longanrou_v3 (温柔闺蜜女)",
        "longhan_v3 (温暖痴情男)",
        "longanzhi_v3 (睿智轻熟男)",
        "longanling_v3 (思维灵动女)",
        "longanya_v3 (高雅气质女)",
        "longanqin_v3 (亲和活泼女)",
        "longjiaxin_v3 (优雅粤语女)",
        "longjiayi_v3 (知性粤语女)",
        "longanyue_v3 (欢脱粤语男)",
        "longlaotie_v3 (东北直率男)",
        "longshange_v3 (原味陕北男)",
        "longanmin_v3 (清纯萝莉/闽南)",
        "loongkyong_v3 (韩语女)",
        "loongriko_v3 (二次元霓虹女)",
        "loongtomoka_v3 (日语女)",
        "longyingxiao_v3 (清甜推销女)",
        "longyingxun_v3 (年轻青涩男)",
        "longyingjing_v3 (低调冷静女)",
        "longyingling_v3 (温和共情女)",
        "longyingtao_v3 (温柔淡定女)",
        "longyingmu_v3 (优雅知性女)",
        "longxiaochun_v3 (知性积极女)",
        "longxiaoxia_v3 (沉稳权威女)",
        "longyumi_v3 (正经青年女)",
        "longanyun_v3 (居家暖男)",
        "longanwen_v3 (优雅知性女)",
        "longanli_v3 (利落从容女)",
        "longanlang_v3 (清爽利落男)",
        "longmiao_v3 (抑扬顿挫女)",
        "longsanshu_v3 (沉稳质感男)",
        "longyuan_v3 (温暖治愈女)",
        "longyue_v3 (温暖磁性女)",
        "longxiu_v3 (博才说书男)",
        "longnan_v3 (睿智青年男)",
        "longwanjun_v3 (细腻柔声女)",
        "longyichen_v3 (洒脱活力男)",
        "longlaobo_v3 (沧桑岁月爷)",
        "longlaoyi_v3 (烟火从容阿姨)",
        "longjiqi_v3 (呆萌机器人)",
        "longhouge_v3 (经典猴哥)",
        "longdaiyu_v3 (娇率才女音)",
        "longanran_v3 (活泼质感女)",
        "longanxuan_v3 (经典直播女)",
        "longshuo_v3 (博才干练男)",
        "longshu_v3 (沉稳青年男)",
        "loongbella_v3 (精准干练女)",
        "longfei_v3 (热血磁性男)",
    ]

    _COSYVOICE_V3_PLUS_OPTIONS = [
        "longanyang (阳光大男孩)",
        "longanhuan (欢脱元气女)",
    ]

    _COSYVOICE_V2_OPTIONS = [
        "longhuhu (天真烂漫女童)",
        "longpaopao (飞天泡泡音)",
        "longjielidou_v2 (阳光顽皮男)",
        "longxian_v2 (豪放可爱女)",
        "longling_v2 (稚气呆板女)",
        "longshanshan (戏剧化童声)",
        "longniuniu (阳光男童声)",
        "longhua_v2 (元气甜美女)",
        "longcheng_v2 (智慧青年男)",
        "longze_v2 (温暖元气男)",
        "longzhe_v2 (呆板大暖男)",
        "longyan_v2 (温暖春风女)",
        "longxing_v2 (温婉邻家女)",
        "longtian_v2 (磁性理智男)",
        "longwan_v2 (细腻柔声女)",
        "longqiang_v2 (浪漫风情女)",
        "longfeifei_v2 (甜美娇气女)",
        "longhao_v2 (多情忧郁男)",
        "longanrou (温柔闺蜜女)",
        "longhan_v2 (温暖痴情男)",
        "longanzhi (睿智轻熟男)",
        "longanling (思维灵动女)",
        "longanya (高雅气质女)",
        "longanqin (亲和活泼女)",
        "longjiayi_v2 (知性粤语女)",
        "longanyue (欢脱粤语男)",
        "longlaotie_v2 (东北直率男)",
        "longshange (原味陕北男)",
        "longanmin (清纯萝莉/闽南)",
        "loongkyong_v2 (韩语女)",
        "loongtomoka_v2 (日语女)",
        "longyingxiao (清甜推销女)",
        "longyingxun (年轻青涩男)",
        "longyingjing (低调冷静女)",
        "longyingling (温和共情女)",
        "longyingtao (温柔淡定女)",
        "longyingmu (优雅知性女)",
        "longxiaochun_v2 (知性积极女)",
        "longxiaoxia_v2 (沉稳权威女)",
        "longyumi_v2 (正经青年女)",
        "longanyun (居家暖男)",
        "longanwen (优雅知性女)",
        "longanli (利落从容女)",
        "longanlang (清爽利落男)",
        "longmiao_v2 (抑扬顿挫女)",
        "longsanshu (沉稳质感男)",
        "longyuan_v2 (温暖治愈女)",
        "longyue_v2 (温暖磁性女)",
        "longxiu_v2 (博才说书男)",
        "longnan_v2 (睿智青年男)",
        "longwanjun (细腻柔声女)",
        "longyichen (洒脱活力男)",
        "longlaobo (沧桑岁月爷)",
        "longlaoyi (烟火从容阿姨)",
        "longjiqi (呆萌机器人)",
        "longhouge (经典猴哥)",
        "longdaiyu (娇率才女音)",
        "longanran (活泼质感女)",
        "longanxuan (经典直播女)",
        "longshuo_v2 (博才干练男)",
        "longshu_v2 (沉稳青年男)",
        "loongbella_v2 (精准干练女)",
        "longfei_v2 (热血磁性男)",
        "longke_v2 (懵懂乖乖女)",
        "longanpei (青少年教师女)",
        "longwangwang (台湾少年音)",
        "longtao_v2 (积极粤语女)",
        "longxiaocheng_v2 (磁性低音男)",
        "longshao_v2 (积极向上男)",
        "kabuleshen_v2 (实力歌手男)",
        "longanshuo (干净清爽男)",
        "longjixin (毒舌心机女)",
        "longgaoseng (得道高僧音)",
        "longyingcui (严肃催收男)",
        "longyingda (开朗高音女)",
        "longyingyan (义正严辞女)",
        "longyingtian (温柔甜美女)",
        "longyingbing (尖锐强势女)",
        "longanchong (激情推销男)",
        "longanping (高亢直播女)",
        "longxiaobai_v2 (沉稳播报女)",
        "longjing_v2 (典型播音女)",
        "loongstella_v2 (飒爽利落女)",
        "libai_v2 (古代诗仙男)",
        "longjin_v2 (优雅温润男)",
        "longbaizhi (睿气旁白女)",
        "loongyuuna_v2 (元气霓虹女)",
        "loongyuuma_v2 (干练霓虹男)",
        "loongtomoya_v2 (日语男)",
        "loongjihun_v2 (阳光韩国男)",
        "loongeva_v2 (知性女)",
        "loongbrian_v2 (沉稳男)",
        "loongluna_v2 (英文女)",
        "loongluca_v2 (英文男)",
        "loongemily_v2 (英文女)",
        "loongeric_v2 (英文男)",
        "loongabby_v2 (英文女)",
        "loongannie_v2 (英文女)",
        "loongandy_v2 (英文男)",
        "loongava_v2 (英文女)",
        "loongbeth_v2 (英文女)",
        "loongbetty_v2 (英文女)",
        "loongcindy_v2 (英文女)",
        "loongcally_v2 (英文女)",
        "loongdavid_v2 (英文男)",
        "loongdonna_v2 (英文女)",
    ]

    _COSYVOICE_V3_FLASH_PARAM = {
        "name": "voice", "label": "音色 (Voice)", "type": "select",
        "options": _COSYVOICE_V3_FLASH_OPTIONS,
        "default": "longanyang",
        "description": "CosyVoice v3 Flash 支持的丰富音色"
    }

    _COSYVOICE_V3_PLUS_PARAM = {
        "name": "voice", "label": "音色 (Voice)", "type": "select",
        "options": _COSYVOICE_V3_PLUS_OPTIONS,
        "default": "longanyang",
        "description": "CosyVoice v3 Plus 高音质音色"
    }

    _COSYVOICE_V2_PARAM = {
        "name": "voice", "label": "音色 (Voice)", "type": "select",
        "options": _COSYVOICE_V2_OPTIONS,
        "default": "longhuhu",
        "description": "CosyVoice v2 多功能音色"
    }

    _LANGUAGE_PARAM = {
        "name": "language_type", "label": "语种 (Language)", "type": "select",
        "options": ["Auto", "Chinese", "English", "Japanese", "Korean", "French", "German"],
        "default": "Auto"
    }

    # --- Video Generation Params ---
    _VIDEO_RESOLUTION_PARAM = {
        "name": "resolution",
        "label": "视频分辨率 (Resolution)",
        "type": "select",
        "default": "1280x720",
        "options": ["1280x720 (720P)", "1920x1080 (1080P)"]
    }
    _VIDEO_DURATION_PARAM = {
        "name": "duration",
        "label": "视频时长 (Duration)",
        "type": "select",
        "default": "5",
        "options": ["5", "10", "15"]
    }
    _VIDEO_PROMPT_EXTEND_PARAM = {
        "name": "prompt_extend",
        "label": "智能改写 (Prompt Extend)",
        "type": "switch",
        "default": True
    }
    _VIDEO_SHOT_TYPE_PARAM = {
        "name": "shot_type",
        "label": "镜头类型 (Shot Type)",
        "type": "select",
        "default": "single",
        "options": ["single (单镜头)", "multi (多镜头)"]
    }
    _VIDEO_AUDIO_PARAM = {
        "name": "generate_audio",
        "label": "生成音效 (Generate Audio)",
        "type": "switch",
        "default": True
    }

    # Omni Model Params
    _OMNI_MODALITIES_PARAM = {
        "name": "modalities", "label": "输出模态 (Modalities)", "type": "select",
        "options": ["text (仅文本)", "text,audio (文本+音频)"],
        "default": "text"
    }

    _OMNI_VOICE_PARAM = {
        "name": "voice", "label": "音色 (Voice - Omni)", "type": "select",
        "options": ["Cherry", "Serena", "Ethan", "Chelsie"],
        "default": "Cherry",
        "description": "仅在不开启思考模式时生效。"
    }

    # --- ASR Params ---
    _DISFLUENCY_REMOVAL_PARAM = {
        "name": "disfluency_removal_enabled", "label": "去口语 (Remove Disfluency)", "type": "switch",
        "default": False,
        "description": "去除‘呃’、‘啊’等语气词。"
    }

    _SPEAKER_DIARIZATION_PARAM = {
        "name": "speaker_diarization_enabled", "label": "区分说话人 (Speaker Diarization)", "type": "switch",
        "default": False,
        "description": "识别不同的说话人 (fun-asr / paraformer识别结果中的 speaker_id)。"
    }

    _ENABLE_ITN_PARAM = {
        "name": "enable_itn", "label": "文本归一化 (ITN)", "type": "switch",
        "default": False,
        "description": "是否启用 ITN (Inverse Text Normalization，逆文本标准化)。"
    }


    MODELS = [
        # --- 0. Omni Models (全模态) ---
        {
            "id": "qwen3.5-plus",
            "name": "Qwen 3.5 Plus (全能专家 - 带推理)",
            "provider": "aliyun",
            "type": "omni",
            "description": "阿里云通义千问 3.5 最新主力全能模型，支持图文影音全模态输入及高深度推理 (Think) 过程，擅长复杂任务分析。",
            "parameters": [_TEMP_PARAM, _TOP_P_PARAM, _TOP_K_PARAM, _REP_PENALTY_PARAM, _SEARCH_PARAM, _THINKING_PARAM, {"name": "modalities", "label": "输出模态", "type": "select", "options": ["text"], "default": "text", "description": "该模型仅支持文本输出"}]
        },
        {
            "id": "qwen3-omni-flash",
            "name": "Qwen3 Omni Flash (全模态)",
            "provider": "aliyun",
            "type": "omni-chat",
            "capabilities": ["chat", "vision", "audio", "video", "search"],
            "description": "支持文本/图片/音频/视频输入，支持文本/音频输出。由阿里云提供。",
            "parameters": [_TEMP_PARAM, _TOP_P_PARAM, _TOP_K_PARAM, _REP_PENALTY_PARAM, _SEARCH_PARAM, _THINKING_PARAM, _OMNI_MODALITIES_PARAM, _OMNI_VOICE_PARAM]
        },

        # --- 1. Text Models (文本) ---
        {
            "id": "qwen-max",
            "name": "通义千问 Max (最强)",
            "provider": "aliyun",
            "type": "text",
            "capabilities": ["chat", "search"],
            "parameters": [_TEMP_PARAM, _TOP_P_PARAM, _TOP_K_PARAM, _REP_PENALTY_PARAM, _SEARCH_PARAM]
        },
        {
            "id": "qwen-plus",
            "name": "通义千问 Plus (均衡)",
            "provider": "aliyun",
            "type": "text",
            "capabilities": ["chat", "search"],
            "parameters": [_TEMP_PARAM, _TOP_P_PARAM, _TOP_K_PARAM, _REP_PENALTY_PARAM, _SEARCH_PARAM, _THINKING_PARAM]
        },
        {
            "id": "qwen-turbo",
            "name": "通义千问 Turbo (极速)",
            "provider": "aliyun",
            "type": "text",
            "capabilities": ["chat", "search"],
            "parameters": [_TEMP_PARAM, _TOP_P_PARAM, _TOP_K_PARAM, _REP_PENALTY_PARAM, _SEARCH_PARAM, _THINKING_PARAM]
        },
        # {
        #     "id": "deepseek-chat",
        #     "name": "DeepSeek V3 (在线)",
        #     "provider": "deepseek",
        #     "type": "text",
        #     "capabilities": ["chat", "code"],
        #     "parameters": [_TEMP_PARAM, _TOP_P_PARAM]
        # },
        {
            "id": "deepseek-v3.2",
            "name": "DeepSeek v3.2 (Aliyun)",
            "provider": "aliyun",
            "type": "text",
            "capabilities": ["chat", "search"],
            "description": "DeepSeek v3.2，支持深度思考 (Reasoning) 模式。",
            "parameters": [_TEMP_PARAM, _TOP_P_PARAM, _TOP_K_PARAM, _REP_PENALTY_PARAM, _SEARCH_PARAM, _THINKING_PARAM]
        },
        {
            "id": "llama3.2:3b",
            "name": "Llama 3.2 3B (本地)",
            "provider": "local",
            "type": "text",
            "capabilities": ["chat"],
            "description": "Meta 最新轻量级模型，运行于本地 Ollama。",
            "parameters": [_TEMP_PARAM, _TOP_P_PARAM, _TOP_K_PARAM, _REP_PENALTY_PARAM, _MAX_TOKENS_PARAM, _SEED_PARAM, _PRESENCE_PENALTY_PARAM, _FREQUENCY_PENALTY_PARAM]
        },

        {
            "id": "qwen2.5:7b",
            "name": "Qwen 2.5 7B (本地)",
            "provider": "local",
            "type": "text",
            "capabilities": ["chat"],
            "description": "通义千问 2.5 7B，最强开源 7B 模型，支持中英双语。",
            "parameters": [_TEMP_PARAM, _TOP_P_PARAM, _TOP_K_PARAM, _REP_PENALTY_PARAM, _MAX_TOKENS_PARAM, _SEED_PARAM, _PRESENCE_PENALTY_PARAM, _FREQUENCY_PENALTY_PARAM]
        },
        {
            "id": "doubao-1-5-pro-32k-250115",
            "name": "豆包 1.5 Pro 32k",
            "provider": "doubao",
            "type": "text",
            "capabilities": ["chat"],
            "description": "豆包纯文本模型，支持调参，最高32k窗口。",
            "parameters": [_TEMP_PARAM, _TOP_P_PARAM, _MAX_TOKENS_PARAM, _FREQUENCY_PENALTY_PARAM]
        },
        {
            "id": "doubao-lite-32k-character-250228",
            "name": "豆包 Lite 32k (角色扮演)",
            "provider": "doubao",
            "type": "text",
            "capabilities": ["chat"],
            "description": "豆包纯文本基础模型，固定参数不可调。",
            "parameters": []
        },
        {
            "id": "doubao-1-5-lite-32k-250115",
            "name": "豆包 1.5 Lite 32k (基础)",
            "provider": "doubao",
            "type": "text",
            "capabilities": ["chat"],
            "description": "豆包纯文本 Lite 模型，固定参数不可调。",
            "parameters": []
        },

        # --- 2. Audio Models (语音) ---
        # {
        #     "id": "sambert-zh-v1",
        #     "name": "Sambert (在线)",
        #     "provider": "aliyun",
        #     "type": "audio-speech", # TTS
        #     "description": "阿里通用语音合成模型。",
        #     "parameters": [
        #         {"name": "voice", "label": "音色", "type": "select", "options": ["zhiya", "guoguo", "zhixiao"], "default": "zhiya"},
        #         _VOICE_SPEED_PARAM
        #     ]
        # },
        # {
        #     "id": "local-chattts",
        #     "name": "ChatTTS (服务器本地)",
        #     "provider": "local",
        #     "type": "audio-speech", # TTS
        #     "description": "适合对话场景的本地语音合成模型。",
        #     "parameters": [_VOICE_SPEED_PARAM]
        # },
        {
            "id": "cosyvoice-v3-flash",
            "name": "CosyVoice v3 Flash (极速版)",
            "provider": "aliyun",
            "type": "audio-speech",
            "description": "CosyVoice 最新极速版，低延迟，适合实时交互。",
            "parameters": [_COSYVOICE_V3_FLASH_PARAM, _VOICE_SPEED_PARAM, _VOLUME_PARAM,  {"name": "format", "label": "音频格式", "type": "select", "options": ["mp3", "wav", "pcm"], "default": "mp3"}]
        },
        {
            "id": "cosyvoice-v3-plus",
            "name": "CosyVoice v3 Plus (高音质)",
            "provider": "aliyun",
            "type": "audio-speech",
            "description": "CosyVoice 高音质版，适合内容创作、有声读物。",
            "parameters": [_COSYVOICE_V3_PLUS_PARAM, _VOICE_SPEED_PARAM, _VOLUME_PARAM,  {"name": "format", "label": "音频格式", "type": "select", "options": ["mp3", "wav", "pcm"], "default": "mp3"}]
        },
        {
            "id": "cosyvoice-v2",
            "name": "CosyVoice v2 (多功能)",
            "provider": "aliyun",
            "type": "audio-speech",
            "description": "支持方言（如粤语）、多情感控制。",
            "parameters": [_COSYVOICE_V2_PARAM, _VOICE_SPEED_PARAM, _VOLUME_PARAM,  {"name": "format", "label": "音频格式", "type": "select", "options": ["mp3", "wav", "pcm"], "default": "mp3"}]
        },


        {
            "id": "qwen3-asr-flash",
            "name": "Qwen3 ASR (通义听悟 - 极速版)",
            "provider": "aliyun",
            "type": "audio-transcription",
            "description": "阿里最新音频大模型，支持极速、实时语音识别。",
            "parameters": [
                _ENABLE_ITN_PARAM
            ]
        },
        {
            "id": "qwen3-asr-flash-filetrans",
            "name": "Qwen3 ASR (通义听悟 - 录音文件识别)",
            "provider": "aliyun",
            "type": "audio-transcription",
            "description": "适合长音频文件转写，支持语义分段。",
            "parameters": [
                 _ENABLE_ITN_PARAM
            ]
        },
        {
            "id": "fun-asr", # Maps to paraformer-v2 or similar internally
            "name": "FunASR (Paraformer - 说话人区分)",
            "provider": "aliyun",
            "type": "audio-transcription",
            "description": "支持说话人区分 (Speaker Diarization) 的语音识别。",
            "parameters": [
                _DISFLUENCY_REMOVAL_PARAM,
                _SPEAKER_DIARIZATION_PARAM
            ]
        },
        {
            "id": "xunfei-lfasr",
            "name": "讯飞语音转写 (Lfasr)",
            "provider": "xunfei",
            "type": "audio-transcription",
            "description": "讯飞长语音转写服务，支持长达 5 小时的音频文件。",
            "parameters": [
                 {"name": "format", "label": "格式", "type": "text", "default": "wav", "description": "支持 wav, mp3 等"}
            ]
        },
        {
            "id": "local-whisper-large",
            "name": "Whisper Large v3 (本地)",
            "provider": "local",
            "type": "audio-transcription",
            "description": "OpenAI Whisper 本地版 (v3 large)，离线可用，支持多语言。",
            "parameters": [
                 _TEMP_PARAM,
                 {"name": "language", "label": "语言 (Language)", "type": "text", "default": None, "description": "目标语言代码 (如 zh, en)，留空自动检测"}
            ]
        },

        # --- 3. Visual Models (视觉) ---
        # {
        #     "id": "wanx-v1",
        #     "name": "通义万相 (在线生成)",
        #     "provider": "aliyun",
        #     "type": "image-generation",
        #     "parameters": [_IMAGE_SIZE_PARAM, _STYLE_PARAM, _NEGATIVE_PROMPT_PARAM, _PROMPT_EXTEND_PARAM]
        # },
        {
            "id": "qwen-image-max",
            "name": "Qwen Image Max (通义万相 - Max)",
            "provider": "aliyun",
            "type": "image-generation",
            "parameters": [_IMAGE_SIZE_PARAM, _NEGATIVE_PROMPT_PARAM, _PROMPT_EXTEND_PARAM]
        },
        {
            "id": "qwen-image-plus",
            "name": "Qwen Image Plus (通义万相 - Plus)",
            "provider": "aliyun",
            "type": "image-generation",
            "parameters": [_IMAGE_SIZE_PARAM, _NEGATIVE_PROMPT_PARAM, _PROMPT_EXTEND_PARAM]
        },
        {
            "id": "doubao-seedream-5-0-260128",
            "name": "豆包 Seedream 5.0 Lite (综合旗舰)",
            "provider": "doubao",
            "type": "image-generation",
            "description": "豆包最新综合视觉旗舰模型。",
            "parameters": [_IMAGE_SIZE_PARAM, {"name": "watermark", "label": "添加水印", "type": "switch", "default": True}, _SEQ_IMG_GEN_PARAM]
        },
        {
            "id": "doubao-seedream-4-5-251128",
            "name": "豆包 Seedream 4.5",
            "provider": "doubao",
            "type": "image-generation",
            "parameters": [_IMAGE_SIZE_PARAM, {"name": "watermark", "label": "添加水印", "type": "switch", "default": False}]
        },
        {
            "id": "doubao-seedream-4-0-250828",
            "name": "豆包 Seedream 4.0",
            "provider": "doubao",
            "type": "image-generation",
            "parameters": [_IMAGE_SIZE_PARAM, {"name": "watermark", "label": "添加水印", "type": "switch", "default": False}, _SEQ_IMG_GEN_PARAM]
        },
        {
            "id": "doubao-seedream-3-0-t2i-250415",
            "name": "豆包 Seedream 3.0 (文生图专用)",
            "provider": "doubao",
            "type": "image-generation",
            "description": "专注于高质量文生图场景。",
            "parameters": [_IMAGE_SIZE_PARAM, _SEED_PARAM]
        },
        # {
        #     "id": "llava",
        #     "name": "LLaVA 7B (本地视觉)",
        #     "provider": "local",
        #     "type": "image-analysis",
        #     "capabilities": ["chat", "vision"],
        #     "description": "本地多模态模型，支持图片理解。运行于 Ollama。",
        #     "parameters": [_TEMP_PARAM, _TOP_P_PARAM, _TOP_K_PARAM, _MAX_TOKENS_PARAM, _SEED_PARAM]
        # },
        {
            "id": "llava-cn",
            "name": "LLaVA 7B 中文版 (本地视觉)",
            "provider": "local",
            "type": "image-analysis",
            "capabilities": ["chat", "vision"],
            "description": "本地多模态模型，支持图片理解，自动翻译为中文输出。",
            "parameters": [_TEMP_PARAM, _TOP_P_PARAM, _TOP_K_PARAM, _MAX_TOKENS_PARAM, _SEED_PARAM]
        },
        {
            "id": "qwen-vl-max",
            "name": "Qwen-VL Max (视觉理解 - Max)",
            "provider": "aliyun",
            "type": "image-analysis", # VQA
            "description": "通义千问视觉理解 Max 版本，能力最强。",
            "parameters": [_TEMP_PARAM, _TOP_P_PARAM]
        },
        {
            "id": "qwen-vl-plus",
            "name": "Qwen-VL Plus (视觉理解 - Plus)",
            "provider": "aliyun",
            "type": "image-analysis", 
            "description": "通义千问视觉理解 Plus 版本，均衡型。",
            "parameters": [_TEMP_PARAM, _TOP_P_PARAM]
        },
        {
            "id": "qwen3-vl-plus",
            "name": "Qwen3-VL Plus (视觉理解3.0 - Plus)",
            "provider": "aliyun",
            "type": "image-analysis",
            "description": "最新 Qwen3 视觉模型。",
            "parameters": [_TEMP_PARAM, _TOP_P_PARAM, _THINKING_PARAM]
        },
        {
            "id": "qwen3-vl-flash",
            "name": "Qwen3-VL Flash (视觉理解3.0 - Flash)",
            "provider": "aliyun",
            "type": "image-analysis",
            "description": "最新 Qwen3 视觉模型，极速响应。",
            "parameters": [_TEMP_PARAM, _TOP_P_PARAM, _THINKING_PARAM]
        },
        # --- 4. Video Models (视频生成) ---
        {
            "id": "wan2.6-i2v-flash",
            "name": "Wanx 2.6 I2V Flash (万相视频 - 极速版)", 
            "provider": "aliyun",
            "type": "video-generation",
            "description": "支持图生视频，多镜头叙事，720P/1080P，时长2-15秒。",
            "parameters": [_VIDEO_RESOLUTION_PARAM, 
                           {"name": "duration", "label": "时长", "type": "slider", "min": 2, "max": 15, "default": 5, "step": 1},
                           _VIDEO_AUDIO_PARAM]
        },
        {
            "id": "wan2.6-i2v", 
            "name": "Wanx 2.6 I2V (万相视频 - 标准版)",
            "provider": "aliyun",
            "type": "video-generation",
            "description": "高质量图生视频，支持5/10/15秒时长。",
            "parameters": [_VIDEO_RESOLUTION_PARAM, _VIDEO_DURATION_PARAM, _VIDEO_AUDIO_PARAM]
        },
        {
            "id": "wan2.6-t2v",
            "name": "Wanx 2.6 T2V (万相文生视频)",
            "provider": "aliyun",
            "type": "video-generation",
            "description": "文生视频，支持声画同步，多镜头叙事。",
            "parameters": [
                           {"name": "resolution", "label": "分辨率", "type": "select", "default": "1920x1080", 
                            "options": [
                                "1920x1080 (1080P 16:9)", "1080x1920 (1080P 9:16)", "1440x1440 (1080P 1:1)", "1632x1248 (1080P 4:3)", "1248x1632 (1080P 3:4)",
                                "1280x720 (720P 16:9)", "720x1280 (720P 9:16)", "960x960 (720P 1:1)", "1088x832 (720P 4:3)", "832x1088 (720P 3:4)"
                            ]},
                           {"name": "duration", "label": "时长", "type": "select", "options": ["5", "10", "15"], "default": "5"},
                           _VIDEO_PROMPT_EXTEND_PARAM,
                           _VIDEO_SHOT_TYPE_PARAM,
                           _VIDEO_AUDIO_PARAM]
        },
        {
            "id": "wan2.5-t2v-preview",
            "name": "Wanx 2.5 T2V Preview (万相文生视频 Preview)",
            "provider": "aliyun",
            "type": "video-generation",
            "description": "文生视频预览版，支持声画同步。",
            "parameters": [
                 {"name": "resolution", "label": "分辨率", "type": "select", "default": "1920x1080",
                  "options": [
                      "1920x1080 (1080P 16:9)", "1080x1920 (1080P 9:16)", "1440x1440 (1080P 1:1)", "1632x1248 (1080P 4:3)", "1248x1632 (1080P 3:4)",
                      "1280x720 (720P 16:9)", "720x1280 (720P 9:16)", "960x960 (720P 1:1)", "1088x832 (720P 4:3)", "832x1088 (720P 3:4)",
                      "832x480 (480P 16:9)", "480x832 (480P 9:16)", "624x624 (480P 1:1)"
                  ]},
                 {"name": "duration", "label": "时长", "type": "select", "options": ["5", "10"], "default": "5"},
                 _VIDEO_PROMPT_EXTEND_PARAM,
                 _VIDEO_AUDIO_PARAM
            ]
        },
        {
            "id": "doubao-seedance-1-5-pro-251215",
            "name": "豆包 Seedance 1.5 Pro (视频生成)",
            "provider": "doubao",
            "type": "video-generation",
            "description": "豆包最新专业级视频生成模型，支持图生视频、文生视频与高级运镜控制。",
            "parameters": [_DOUBAO_VIDEO_RESOLUTION_PARAM, _DOUBAO_VIDEO_RATIO_PARAM, _DOUBAO_VIDEO_DURATION_PARAM, _VIDEO_CAMERA_FIXED_PARAM, _VIDEO_WATERMARK_PARAM, _SEED_PARAM, _VIDEO_AUDIO_PARAM]
        },
        {
            "id": "doubao-seedance-1-0-pro-250528",
            "name": "豆包 Seedance 1.0 Pro",
            "provider": "doubao",
            "type": "video-generation",
            "description": "高质量图生视频/文生视频，1080p高清支持。",
            "parameters": [_DOUBAO_VIDEO_RESOLUTION_PARAM, _DOUBAO_VIDEO_RATIO_PARAM, _DOUBAO_VIDEO_DURATION_PARAM, _VIDEO_CAMERA_FIXED_PARAM, _VIDEO_WATERMARK_PARAM, _SEED_PARAM]
        },
        {
            "id": "doubao-seedance-1-0-pro-fast-251015",
            "name": "豆包 Seedance 1.0 Pro Fast",
            "provider": "doubao",
            "type": "video-generation",
            "description": "极速版本，更快的生成速度。",
            "parameters": [_DOUBAO_VIDEO_RESOLUTION_PARAM, _DOUBAO_VIDEO_RATIO_PARAM, _DOUBAO_VIDEO_DURATION_PARAM, _VIDEO_CAMERA_FIXED_PARAM, _VIDEO_WATERMARK_PARAM, _SEED_PARAM]
        },
        {
            "id": "doubao-seedance-1-0-lite-i2v-250428",
            "name": "豆包 Seedance 1.0 Lite I2V (图生视频专用)",
            "provider": "doubao",
            "type": "video-generation",
            "description": "轻量级图生视频，支持参考图与首尾帧生成。",
            "parameters": [_DOUBAO_VIDEO_RESOLUTION_PARAM, _DOUBAO_VIDEO_RATIO_PARAM, _DOUBAO_VIDEO_DURATION_PARAM, _VIDEO_WATERMARK_PARAM, _SEED_PARAM]
        },
        {
            "id": "doubao-seedance-1-0-lite-t2v-250428",
            "name": "豆包 Seedance 1.0 Lite T2V (文生视频专用)",
            "provider": "doubao",
            "type": "video-generation",
            "description": "轻量级文生视频。",
            "parameters": [_DOUBAO_VIDEO_RESOLUTION_PARAM, _DOUBAO_VIDEO_RATIO_PARAM, _DOUBAO_VIDEO_DURATION_PARAM, _VIDEO_CAMERA_FIXED_PARAM, _VIDEO_WATERMARK_PARAM, _SEED_PARAM]
        },
        # --- 4. Multimodal Models (多模态) ---
        {
            "id": "doubao-seed-1-8-251228",
            "name": "豆包 Seed 1.8 (全模态)",
            "provider": "doubao",
            "type": "omni",
            "description": "豆包多模态系列前代精良模型，支持文本与视觉交互。",
            "capabilities": ["chat", "vision", "video"],
            "parameters": [_TEMP_PARAM, _TOP_P_PARAM, _THINKING_PARAM, _REASONING_EFFORT_PARAM, _MAX_TOKENS_PARAM]
        },
        {
            "id": "doubao-seed-2-0-pro-260215",
            "name": "豆包 Seed 2.0 Pro (全模态/推理)",
            "provider": "doubao",
            "type": "omni",
            "description": "豆包最新旗舰版大模型，支持长上下文、图片视频理解与高强度推理。",
            "capabilities": ["chat", "vision", "video"],
            "parameters": [_FIXED_TEMP_PARAM, _FIXED_TOP_P_PARAM, _THINKING_PARAM, _REASONING_EFFORT_PARAM, _MAX_TOKENS_PARAM]
        },
        {
            "id": "doubao-seed-2-0-mini-260215",
            "name": "豆包 Seed 2.0 Mini (全模态)",
            "provider": "doubao",
            "type": "omni",
            "description": "豆包高性能轻量版模型，极速响应，支持多模态理解。",
            "capabilities": ["chat", "vision", "video"],
            "parameters": [_TEMP_PARAM, _TOP_P_PARAM, _THINKING_PARAM, _REASONING_EFFORT_PARAM, _MAX_TOKENS_PARAM]
        },
        {
            "id": "doubao-seed-1-6-251015",
            "name": "豆包 Seed 1.6 (全模态)",
            "provider": "doubao",
            "type": "omni",
            "description": "豆包经典主力模型，支持文本与视觉多模态交互。",
            "capabilities": ["chat", "vision", "video"],
            "parameters": [_TEMP_PARAM, _TOP_P_PARAM, _THINKING_PARAM, _MAX_TOKENS_PARAM]
        },
        {
            "id": "doubao-seed-1-6-flash-250828",
            "name": "豆包 Seed 1.6 Flash (全模态)",
            "provider": "doubao",
            "type": "omni",
            "description": "豆包极速版大模型，适合低延迟场景的多模态分析。",
            "capabilities": ["chat", "vision", "video"],
            "parameters": [_TEMP_PARAM, _TOP_P_PARAM, _THINKING_PARAM, _MAX_TOKENS_PARAM]
        },
        # --- 5. Image Generation Models (图片生成) ---
        {
            "id": "doubao-seedream-5-0-260128",
            "name": "豆包 Seedream 5.0 Lite (综合旗舰)",
            "provider": "doubao",
            "type": "image",
            "description": "豆包图像生成旗舰模型，支持文生图，及最多14张单/多图生组图。自带联网搜索能力。",
            "capabilities": ["image_generation", "vision"],
            "parameters": [_IMAGE_SIZE_PARAM, _SEQ_IMG_GEN_PARAM, _WATERMARK_PARAM]
        },
        {
            "id": "doubao-seedream-4-5-251128",
            "name": "豆包 Seedream 4.5",
            "provider": "doubao",
            "type": "image",
            "description": "豆包图像生成进阶模型，支持文生图与图生组图。",
            "capabilities": ["image_generation", "vision"],
            "parameters": [_IMAGE_SIZE_PARAM, _SEQ_IMG_GEN_PARAM, _WATERMARK_PARAM]
        },
        {
            "id": "doubao-seedream-4-0-250828",
            "name": "豆包 Seedream 4.0",
            "provider": "doubao",
            "type": "image",
            "description": "豆包图像生成主流模型。",
            "capabilities": ["image_generation", "vision"],
            "parameters": [_IMAGE_SIZE_PARAM, _SEQ_IMG_GEN_PARAM, _WATERMARK_PARAM]
        },
        {
            "id": "doubao-seedream-3-0-t2i-250415",
            "name": "豆包 Seedream 3.0 (文生图专用)",
            "provider": "doubao",
            "type": "image",
            "description": "纯纯的文生单图模型。此模型必须使用 seed 和 guidance 限制进行约束，不支持组图和垫图。",
            "capabilities": ["image_generation"],
            "parameters": [_IMAGE_SIZE_PARAM, _SEED_PARAM, _WATERMARK_PARAM]
        },
        # --- 6. Video Generation Models (视频生成) ---
        {
            "id": "doubao-seedance-1-5-pro-251215",
            "name": "豆包 Seedance 1.5 Pro (综合旗舰)",
            "provider": "doubao",
            "type": "video",
            "description": "豆包强大的视频生成模型，支持文生视频及图生视频（含首尾帧），可自选音频生成。",
            "capabilities": ["video_generation", "vision"],
            "parameters": [_VIDEO_RESOLUTION_PARAM, _VIDEO_RATIO_PARAM, _VIDEO_DURATION_PARAM, _SEED_PARAM, _VIDEO_CAMERA_FIXED_PARAM, _VIDEO_WATERMARK_PARAM]
        },
        {
            "id": "doubao-seedance-1-0-pro-250528",
            "name": "豆包 Seedance 1.0 Pro",
            "provider": "doubao",
            "type": "video",
            "description": "豆包视频生成旗舰模型。",
            "capabilities": ["video_generation", "vision"],
            "parameters": [_VIDEO_RESOLUTION_PARAM, _VIDEO_RATIO_PARAM, _VIDEO_DURATION_PARAM, _SEED_PARAM, _VIDEO_CAMERA_FIXED_PARAM, _VIDEO_WATERMARK_PARAM]
        },
        {
            "id": "doubao-seedance-1-0-pro-fast-251015",
            "name": "豆包 Seedance 1.0 Pro Fast",
            "provider": "doubao",
            "type": "video",
            "description": "豆包视频生成旗舰极速版，消耗成本较低。",
            "capabilities": ["video_generation", "vision"],
            "parameters": [_VIDEO_RESOLUTION_PARAM, _VIDEO_RATIO_PARAM, _VIDEO_DURATION_PARAM, _SEED_PARAM, _VIDEO_CAMERA_FIXED_PARAM, _VIDEO_WATERMARK_PARAM]
        },
        {
            "id": "doubao-seedance-1-0-lite-i2v-250428",
            "name": "豆包 Seedance 1.0 Lite (图生视频)",
            "provider": "doubao",
            "type": "video",
            "description": "豆包视频生成图生视频专属版本，支持输入多张参考图进行生成。",
            "capabilities": ["video_generation", "vision"],
            "parameters": [_VIDEO_RESOLUTION_PARAM, _VIDEO_RATIO_PARAM, _VIDEO_DURATION_PARAM, _SEED_PARAM, _VIDEO_WATERMARK_PARAM]
        },
        {
            "id": "doubao-seedance-1-0-lite-t2v-250428",
            "name": "豆包 Seedance 1.0 Lite (文生视频)",
            "provider": "doubao",
            "type": "video",
            "description": "豆包视频生成文生视频专属轻量版本。",
            "capabilities": ["video_generation"],
            "parameters": [_VIDEO_RESOLUTION_PARAM, _VIDEO_RATIO_PARAM, _VIDEO_DURATION_PARAM, _SEED_PARAM, _VIDEO_CAMERA_FIXED_PARAM, _VIDEO_WATERMARK_PARAM]
        }
    ]

    @classmethod
    def get_all_models(cls) -> List[Dict[str, Any]]:
        return cls.MODELS

    @classmethod
    def get_model(cls, model_id: str) -> Union[Dict[str, Any], None]:
        for model in cls.MODELS:
            if model["id"] == model_id:
                return model
        return None
