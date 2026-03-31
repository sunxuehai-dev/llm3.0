from ..LlamaConfig import LlamaConfig


class Train:
    def __init__(self, config: LlamaConfig) -> None:
        """
        初始化训练配置
        :param config: LlamaConfig实例，包含所有配置参数
        该类用于设置训练相关参数
        """
        self._config = config
    
    def set_learning_rate(self, lr: float = 1e-4):
        """
        设置学习率
        :param lr: 学习率值
        该函数设置config中的learning_rate参数
        """
        self._config.learning_rate = lr

    def set_bf16(self, flag: bool = True):
        """
        设置是否使用BF16混合精度训练
        :param flag: 是否启用BF16，默认True
        该函数设置config中的bf16参数
        """
        self._config.bf16 = flag

    def set_resume_from_checkpoint(self, method: str = 'null'):
        """
        设置从检查点恢复训练的方法
        :param method: 检查点路径或恢复方法
        该函数设置config中的resume_from_checkpoint参数
        """
        self._config.resume_from_checkpoint = method

    def set_num_train_epochs(self, epochs: float = 3.0):
        """
        设置训练轮数
        :param epochs: 训练轮数
        该函数设置config中的num_train_epochs参数
        """
        self._config.num_train_epochs = epochs

    def set_per_device_train_batch_size(self, batch_size: int = 1):
        """
        设置每个设备的训练批次大小
        :param batch_size: 每个设备的批次大小
        该函数设置config中的per_device_train_batch_size参数
        """
        self._config.per_device_train_batch_size = batch_size

    def set_gradient_accumulation_steps(self, steps: int):
        """
        设置梯度累积步数
        :param steps: 梯度累积的步数
        该函数设置config中的gradient_accumulation_steps参数
        """
        self._config.gradient_accumulation_steps = steps

    def set_lr_scheduler_type(self, scheduler_type: str = 'cosine'):
        """
        设置学习率调度器类型
        :param scheduler_type: 调度器类型名称
        该函数设置config中的lr_scheduler_type参数
        """
        self._config.lr_scheduler_type = scheduler_type

    def set_warmup_ratio(self, ratio: float = 0.1):
        """
        设置学习率预热比例
        :param ratio: 预热步数占总步数的比例
        该函数设置config中的warmup_ratio参数
        """
        self._config.warmup_ratio = ratio

    def set_ddp_timeout(self, timeout: int = 180000000):
        """
        设置DDP通信超时时间（毫秒）
        :param timeout: 超时时间值
        该函数设置config中的ddp_timeout参数
        """
        self._config.ddp_timeout = timeout
