from ..LlamaConfig import LlamaConfig
from typing import override


class AbstractFineTuning:
    TYPES = ['full', 'lora', 'qlora']

    def __init__(self, ft_type) -> None:
        self._finetuning_type = ft_type

    def get_type(self) -> str:
        return self._finetuning_type

    def active_config(self, config: LlamaConfig):
        config.finetuning_type = self._finetuning_type


class FullFineTuning(AbstractFineTuning):
    def __init__(self, finetuning_type) -> None:
        super().__init__(finetuning_type)

    @override
    def active_config(self, config: LlamaConfig) -> None:
        super().active_config(config)


class LoraFineTuning(AbstractFineTuning):
    def __init__(self, finetuning_type) -> None:
        super().__init__(finetuning_type)
        self._rank: int = 0
        self._target: str = ''

    def set_rank(self, rank: int) -> None:
        self._rank = rank

    def set_target(self, target: str) -> None:
        self._target = target

    @override
    def active_config(self, config: LlamaConfig) -> None:
        super().active_config(config)
        config.lora_rank = self._rank
        config.lora_target = self._target
