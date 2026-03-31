from ..LlamaConfig import LlamaConfig
from functools import wraps
from pathlib import Path
from .StageType import *
from typing import Optional

def _StageType(stage_type: str) -> Optional[AbstractStageType]:
    if stage_type not in AbstractStageType.TYPES:
        raise ValueError(f"Invalid finetuning_type: {stage_type}. Valid options are: {', '.join(AbstractStageType.TYPES)}")

    if stage_type == 'dpo':
        return DpoStage('dpo')
    elif stage_type == 'kto':
        return KtoStage('kto')
    elif stage_type == 'pt':
        return PtStage('pt')
    elif stage_type == 'rm':
        return RewardStage('rm')
    elif stage_type == 'sft':
        return SftStage('sft')


def _handle_stage_task(stage_type: str = ''):
    def decorator(task_func):
        @wraps(task_func)
        def wrapper(self, *args, **kwargs):
            entity = _StageType(stage_type)

            self._entity = entity
            task_func(self, *args, **kwargs)

            if isinstance(entity, AbstractStageType):
                entity.active_config(self._config)

        return wrapper
    return decorator


class SetStage:
    def __init__(self, config: LlamaConfig) -> None:
        self._config = config
        self._entity = None

    @_handle_stage_task('dpo')
    def set_dpo(self, pref_beta: float, pref_loss: str = 'sigmoid', do_train: bool = True) -> None:
        entity = self._entity
        if isinstance(entity, DpoStage):
            entity.set_do_train(do_train)
            entity.set_pref_beta(pref_beta)
            entity.set_pref_loss(pref_loss)

    @_handle_stage_task('kto')
    def set_kto(self, pref_beta: float, do_train: bool = True) -> None:
        entity = self._entity
        if isinstance(entity, KtoStage):
            entity.set_do_train(do_train)
            entity.set_pref_beta(pref_beta)

    @_handle_stage_task('pt')
    def set_pretrain(self, do_train: bool = True) -> None:
        entity = self._entity
        if isinstance(entity, PtStage):
            entity.set_do_train(do_train)

    @_handle_stage_task('rm')
    def set_reward(self, do_train: bool = True) -> None:
        entity = self._entity
        if isinstance(entity, RewardStage):
            entity.set_do_train(do_train)
            entity.set_gradient_accumulation_steps()

    @_handle_stage_task('sft')
    def set_sft(self, deepspeed: Optional[Path] = None, do_train: bool = True) -> None:
        entity = self._entity
        if isinstance(entity, SftStage):
            entity.set_do_train(do_train)
            if deepspeed is not None:
                entity.set_deepspeed(deepspeed)
