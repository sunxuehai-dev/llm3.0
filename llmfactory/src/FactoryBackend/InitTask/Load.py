from pathlib import Path
from typing import Optional
from ..LlamaConfig import LlamaConfig


class InitDataset:
    def __init__(self, config: LlamaConfig, *,
                dataset: str = '',  # hugginface remote
                dataset_dir: Path = Path(), # 本地路径
                ) -> None:
        """
        初始化数据集配置
        :param config: LlamaConfig实例，包含所有配置参数
        :param dataset: Hugging Face数据集名称（远程）
        :param dataset_dir: 本地数据集目录路径
        该函数设置config中的dataset和dataset_dir参数
        """
        self._config = config
        if dataset != '':
            self._config.dataset = dataset

        if dataset_dir != Path():
            self._config.dataset_dir = str(dataset_dir.resolve())


    def set_cutoff_len(self, cutoff_len: int = 2048):
        """
        设置输入序列的最大长度
        :param cutoff_len: 输入序列的最大长度（token数），默认2048
        该函数设置config中的cutoff_len参数
        """
        self._config.cutoff_len = cutoff_len

    def set_max_samples(self, maxsample: int = 1000):
        """
        设置训练中使用的最大样本数量
        :param maxsample: 最大样本数，默认1000
        该函数设置config中的max_samples参数
        """
        self._config.max_samples = maxsample


class InitModel:

    template_register = [
        '-',
        'deepseek',
        'deepseek3',
        'deepseekr1',
        'ernie_nothink',
        'falcon',
        'falcon_h1',
        'gemma',
        'gemma2',
        'gemma3',
        'gemma3n',
        'glm4',
        'glmz1',
        'glm4_moe',
        'glm4_5v',
        'gpt_oss',
        'granite3',
        'granite4',
        'hunyuan',
        'hunyuan_small',
        'intern2',
        'intern_vl',
        'intern_s1',
        'kimi_vl',
        'bailing_v2',
        'lfm2',
        'lfm2_vl',
        'llama2',
        'llama3',
        'llama4',
        'mllama',
        'llava',
        'llava_next',
        'llava_next_video',
        'mimo',
        'mimo_v2',
        'cpm4',
        'minicpm_o',
        'minicpm_v',
        'minimax1',
        'minimax2',
        'ministral3',
        'mistral',
        'paligemma',
        'phi',
        'phi_small',
        'phi4_mini',
        'phi4',
        'pixtral',
        'qwen',
        'qwen3',
        'qwen3_nothink',
        'qwen2_audio',
        'qwen2_omni',
        'qwen3_omni',
        'qwen2_vl',
        'qwen3_vl',
        'seed_oss',
        'seed_coder',
        'yuan'
    ]

    def __init__(self, config: LlamaConfig, *,
                model_name: str = '',  # hugginface remote
                model_path: Path = Path(), # 本地路径
                trust_remote_code: bool = True) -> None:
        """
        初始化模型配置
        :param config: LlamaConfig实例，包含所有配置参数
        :param model_name: Hugging Face模型名称或路径（远程）
        :param model_path: 本地模型路径
        :param trust_remote_code: 是否信任远程代码，默认为True
        该函数设置config中的model_name_or_path参数
        """
        self._config = config
        if model_name == '' and model_path == Path():
            raise ValueError("model_name or model_path must be set")

        if model_name != '':
            self._config.model_name_or_path = model_name
        elif model_path != Path() and model_path.exists():
            self._config.model_name_or_path = str(model_path.resolve())
        else:
            raise ValueError("model_path not exists!")

        self._trust_remote_code: bool = trust_remote_code

    def set_template(self, template: str):
        """
        设置模型和数据集的模板
        :param template: 模板名称，需要和模型对应
        该函数设置config中的template参数
        参考: https://github.com/hiyouga/LlamaFactory/blob/main/README_zh.md
        """
        if template not in self.template_register:
            raise ValueError("template could not be found! plaease read: https://github.com/hiyouga/LlamaFactory/blob/main/README_zh.md")
        self._config.template = template

    def set_adapter_name_or_path(self, adapter_name_or_path: Path):
        if adapter_name_or_path.exists():
            self._config.adapter_name_or_path = str(adapter_name_or_path.resolve())
        else:
            raise ValueError("adapter_name_or_path not exists!")

    def enbale_VL(self, image_max_pixels: int = 262144, video_max_pixels: int = 16384):
        """
        启用视觉语言（Vision-Language）模型的支持
        :param image_max_pixels: 图像最大像素数，默认262144
        :param video_max_pixels: 视频最大像素数，默认16384
        该函数设置config中的image_max_pixels和video_max_pixels参数
        """ 
        self._config.image_max_pixels = image_max_pixels
        self._config.video_max_pixels = video_max_pixels

    def enable_quantized(self, method: str, bit: int):
        """
        Method: [bnb, hqq, eetq]
        Bit: [8 (bnb/hqq/eetq), 4 (bnb/hqq), 3 (hqq), 2 (hqq)]
        """
        self._config.quantization_bit = bit
        self._config.quantization_method = method

