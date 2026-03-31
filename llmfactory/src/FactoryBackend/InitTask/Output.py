from ..LlamaConfig import LlamaConfig
from pathlib import Path


class Output:
    def __init__(self, config: LlamaConfig) -> None:
        self._config = config
        
    def set_output_dir(self, path: Path):
        """
        设置模型输出目录
        :param path: 模型输出目录路径
        该函数设置config中的output_dir参数，用于保存训练过程中生成的模型权重和日志文件
        """
        self._config.output_dir = str(path.resolve())

    def set_logging_steps(self, logging_steps: int = 10):
        """
        设置训练的步数
        :param steps: 训练的步数，默认10
        该函数设置config中的logging_steps参数，控制训练过程中日志的输出间隔
        """
        self._config.logging_steps = logging_steps

    def set_save_steps(self, steps: int = 500):
        """
        设置训练的步数
        :param steps: 训练的步数，默认100
        该函数设置config中的num_train_steps参数，控制训练的步数
        """
        self._config.save_steps = steps

    def set_overwrite_output_dir(self, overwrite: bool = True):
        """
        设置是否覆盖输出目录
        :param overwrite: 是否覆盖已存在的输出目录，默认为True
        该函数设置config中的overwrite_output_dir参数，避免训练中断
        """
        self._config.overwrite_output_dir = overwrite

    def set_save_only_model(self, save_only: bool = True):
        """
        设置是否仅保存模型权重
        :param save_only: True仅保存模型权重，False保存完整检查点（含优化器状态）
        该函数设置config中的save_only_model参数
        """
        self._config.save_only_model = save_only

    def set_plot_loss(self, plot: bool = True):
        """
        设置是否生成训练损失曲线图
        :param plot: True生成损失图，False不生成
        该函数设置config中的plot_loss参数，便于分析收敛情况
        """
        self._config.plot_loss = plot

    def set_report_to(self, platform: str = 'none'):
        """
        设置训练监控报告平台
        :param platform: 报告平台，可选值：'none', 'wandb', 'tensorboard', 'swanlab', 'mlflow'
        该函数设置config中的report_to参数，用于配置训练过程的监控和日志记录平台
        """
        valid_platforms = ['none', 'wandb', 'tensorboard', 'swanlab', 'mlflow']
        if platform not in valid_platforms:
            raise ValueError(f"Invalid report_to value: {platform}. Must be one of {valid_platforms}")
        self._config.report_to = platform


class Export:
    def __init__(self, config: LlamaConfig) -> None:
        self._config = config
        
    def set_export_dir(self, path: Path):
        """
        设置模型导出目录
        :param path: 导出目录路径
        该函数设置config中的export_dir参数，用于保存合并后的完整模型
        """
        self._config.export_dir = str(path.resolve())

    def set_export_size(self, size: int = 5):
        """
        设置单个导出文件的大小（以GB为单位）
        :param size: 每个导出文件的大小（GB），默认为1GB
        该函数设置config中的export_size参数，控制分片导出文件的大小
        """
        self._config.export_size = size

    def set_export_legacy_format(self, legacy_format: bool = False):
        """
        设置是否使用旧版导出格式
        :param legacy_format: True使用旧版格式，False使用新版格式，默认为False
        该函数设置config中的export_legacy_format参数
        """
        self._config.export_legacy_format = legacy_format

    def set_export_quantization_bit(self, bit: int = 4):
        """
        设置导出模型的量化位数
        :param bit: 量化位数（如4或8），None表示不进行量化
        该函数设置config中的export_quantization_bit参数，用于导出量化模型
        """
        self._config.export_quantization_bit = bit

    def set_export_quantization_dataset(self, dataset_path: Path):
        """
        设置用于量化校准的数据集
        :param dataset_path: 量化校准数据集路径
        该函数设置config中的export_quantization_dataset参数，用于模型量化时的校准数据
        """
        self._config.export_quantization_dataset = str(dataset_path.resolve())
    
    def set_export_device(self, choice: str = 'auto'):
        """
        [cpu, auto]
        """
        self._config.export_device = choice