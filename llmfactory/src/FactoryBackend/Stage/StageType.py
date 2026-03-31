from ..LlamaConfig import LlamaConfig
from pathlib import Path
from typing import override

class AbstractStageType:
    TYPES = [
        'dpo', 
        'kto', 
        'pt',   # Method: 'pretrain'
        'rm',   # Method: 'reward'
        'sft',  # sft can enable 'ds3, ray'
    ]

    def __init__(self, stage) -> None:
        self._stage = stage
        self._do_train = None

    def get_type(self) -> str:
        return self._stage

    def set_do_train(self, do_train: bool) -> None:
        self._do_train = do_train    

    def active_config(self, config: LlamaConfig) -> None:
        config.stage = self._stage
        config.do_train = self._do_train


class DpoStage(AbstractStageType):
    def __init__(self, stage) -> None:
        super().__init__(stage)
        self._pref_beta = None
        self._pref_loss = None

    def set_pref_beta(self, pref_beta: float):
        self._pref_beta = pref_beta

    def set_pref_loss(self, pref_loss: str):
        """
        pref_loss choices: [sigmoid (dpo), orpo, simpo]
        """
        self._pref_loss = pref_loss

    @override
    def active_config(self, config: LlamaConfig):
        super().active_config(config)
        config.pref_beta = self._pref_beta
        config.pref_loss = self._pref_loss


class KtoStage(AbstractStageType):
    def __init__(self, stage) -> None:
        super().__init__(stage)
        self._pref_beta = None

    def set_pref_beta(self, pref_beta: float):
        self._pref_beta = pref_beta

    @override
    def active_config(self, config: LlamaConfig):
        super().active_config(config)
        config.pref_beta = self._pref_beta


class PtStage(AbstractStageType):
    """
    Method: pretrain
    """
    def __init__(self, stage) -> None:
        super().__init__(stage)


class RewardStage(AbstractStageType):
    def __init__(self, stage) -> None:
        super().__init__(stage)
        self._gradient_accumulation_steps = None

    def set_gradient_accumulation_steps(self, steps: int = 8):
        self._gradient_accumulation_steps = steps

    @override
    def active_config(self, config: LlamaConfig):
        super().active_config(config)
        config.gradient_accumulation_steps = self._gradient_accumulation_steps


class SftStage(AbstractStageType):
    def __init__(self, stage) -> None:
        super().__init__(stage)

        self._deepspeed = None

    def set_do_train(self, do_train: bool):
        self._do_train = do_train

    def set_deepspeed(self, config_file_path: Path):
        self._deepspeed = str(config_file_path.resolve())
    
    @override
    def active_config(self, config: LlamaConfig):
        super().active_config(config)
        config.do_train = self._do_train
        config.deepspeed = self._deepspeed