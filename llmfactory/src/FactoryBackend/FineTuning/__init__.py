from typing import Optional
from functools import wraps

from ..LlamaConfig import LlamaConfig
from .FineTuningType import AbstractFineTuning
from .FineTuningType import LoraFineTuning
from .FineTuningType import FullFineTuning


def _FinetuningType(finetuning_type: str) -> Optional[AbstractFineTuning]:
    if finetuning_type not in AbstractFineTuning.TYPES:
        raise ValueError(f"Invalid finetuning_type: {finetuning_type}. Valid options are: {', '.join(AbstractFineTuning.TYPES)}")
    
    if finetuning_type == 'lora':
        return LoraFineTuning('lora')
    elif finetuning_type == 'full':
        return FullFineTuning('full')


def _handle_finetuning_task(finetuning_type: str = ''):
    def decorator(task_func):
        @wraps(task_func)
        def wrapper(self, *args, **kwargs):
            entity = _FinetuningType(finetuning_type)

            self._entity = entity
            task_func(self, *args, **kwargs)
            
            if isinstance(entity, AbstractFineTuning):
                entity.active_config(self._config)
        
        return wrapper
    return decorator


class SetFinetuning:
    def __init__(self, config: LlamaConfig) -> None:
        self._config = config
        self._entity = None

    @_handle_finetuning_task('lora')
    def set_lora(self, rank: int, target: str) -> None:
        entity = self._entity
        if isinstance(entity, LoraFineTuning):
            entity.set_rank(rank)
            entity.set_target(target)
    
    @_handle_finetuning_task('full')
    def set_full(self) -> None:
        pass