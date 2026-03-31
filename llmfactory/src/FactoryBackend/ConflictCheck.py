from signal import raise_signal
from .LlamaConfig import LlamaConfig

def _check_model_name_or_path(config):
    if config.model_name_or_path is None:
        raise ValueError(f"Configuration is error: model_name_or_path is not set.")

def _check_dataset(config):
    if config.dataset is None and config.dataset_dir is None:
        raise ValueError(f"Configuration is conflicting: dataset or dataset_dir is not set.")

def _check_template(config):
    if config.model_name_or_path and config.template is None:
        raise ValueError(f"Configuration is conflicting: model_name_or_path is set, but template is not set.\n Please read: https://github.com/hiyouga/LlamaFactory/blob/main/README_zh.md")

def _check_export_dir(config):
    if config.export_dir and config.adapter_name_or_path is None and config.finetuning_type == 'lora':
        raise ValueError(f"Configuration is conflicting: export_dir is set, but adapter_name_or_path is not set for lora fine-tuning.")

def _check_deepspeed(config):
    if config.stage != 'sft' and config.deepspeed is not None:
        raise ValueError(f"Configuration is conflicting: deepspeed is set, but stage is not 'sft'(sft).")

def _check_pre_beta(config):
    if config.stage not in ['kto', 'dpo'] and config.pref_beta is not None:
        raise ValueError(f"Configuration is conflicting: pre_beta is set, but stage is not 'kto'(kto).")

def _check_pre_loss(config):
    if config.stage != 'dpo' and config.pref_loss is not None:
        raise ValueError(f"Configuration is conflicting: pre_loss is set, but stage is not 'dpo'(dpo).")

def _check_lora(config):
    if config.finetuning_type != 'lora' and (config.lora_rank is not None or config.lora_target is not None):
        raise ValueError(f"Configuration is conflicting: lora_rank or lora_target is set, but finetuning_type is not 'lora'.")

def _check_distributed(config):
    if config.deepspeed is not None:
        if config.gradient_accumulation_steps is None:
            raise ValueError(f"Configuration is conflicting: Distributed enable!, but gradient_accumulation_steps is not set.")

def _check_galore(config):
    if config.use_galore is not None and config.deepspeed is not None:
        raise ValueError(f"Configuration is conflicting: deepseek can't support galore.")

def _check_qlora(config):
    if config.quantization_method not in ['bnb', 'hqq', 'eetq']:
        raise ValueError(f"{config.quantization_method} is not in ['bnb', 'hqq', 'eetq'] ")

    if config.quantization_bit not in [8, 4, 3, 2]:
        raise ValueError(f"{config.quantization_bit} is not in [8, 4, 3, 2]")

    if config.quantization_bit == 4:
        if config.quantization_method == 'eetq':
            raise ValueError(
                f"Method: {config.quantization_method} conflict with Bit: {config.quantization_bit}\n choices: [8 (bnb/hqq/eetq), 4 (bnb/hqq), 3 (hqq), 2 (hqq)]")

    if config.quantization_bit in [2, 3]:
        if config.quantization_method != 'hqq':
            raise ValueError(
                f"Method: {config.quantization_method} conflict with Bit: {config.quantization_bit}\n choices: [8 (bnb/hqq/eetq), 4 (bnb/hqq), 3 (hqq), 2 (hqq)]")


class ConfigValidator:
    def __init__(self, config: LlamaConfig, command_type = 'train'):
        self._config = config
        self._command_type = command_type

        self.checks = [
            _check_model_name_or_path,
            _check_dataset,
            _check_template,
            _check_export_dir,
            _check_deepspeed,
            _check_pre_beta,
            _check_pre_loss,
            _check_lora,
            _check_distributed,
            _check_galore
        ]

    def validate(self):
        if self._command_type == 'train':
            for check in self.checks:
                check(self._config)
