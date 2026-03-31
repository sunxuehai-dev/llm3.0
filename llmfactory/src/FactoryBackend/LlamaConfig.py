from dataclasses import dataclass
from typing import Optional


@dataclass
class LlamaConfig:
    model_name_or_path:      Optional[str]  = None
    adapter_name_or_path:    Optional[str]  = None
    infer_backend:           Optional[str] = None
    trust_remote_code:       Optional[bool] = True   # 允许对非典型架构模型运行，此处默认开

    # ======= For VL Model=========
    image_max_pixels:        Optional[int]  = None
    video_max_pixels:        Optional[int]  = None
    #==============================

    # ======= For Qlora ===========
    quantization_bit:        Optional[int] = None
    quantization_method:     Optional[str] = None    # [bnb, hqq, eetq]


    # ========== Method ===========
    stage:                   Optional[str] = None    # 训练阶段方式
    do_train:                Optional[bool] = None   # 默认应该设为 True， 启用训练模式
    finetuning_type:         Optional[str] = None    # 微调方式

    # if finetuning_type == 'lora':
    lora_rank:               Optional[int] = None
    lora_target:             Optional[str] = None

    # if stage is: kto
    pref_beta:                Optional[float] = None # default 0.1
    # if stage is dpo
    pref_loss:                Optional[str] = None

    # if enable deepspeed
    deepspeed:               Optional[str] = None    # deepseed config file path

    # if enable galore
    use_galore:              Optional[bool] = None
    galore_layerwise:        Optional[bool] = None  # choices: [true, false], use false for DDP training
    galore_target:           Optional[str] = None
    galore_rank:             Optional[int] = None
    galore_scale:            Optional[float] = None
    optim:                   Optional[str] = None
    # =============================

    # ========== Dataset ==========
    dataset:                 Optional[str] = None    # huggingface model
    dataset_dir:             Optional[str] = None    # use local absolute path
    template:                Optional[str] = None    # model & template map: https://github.com/hiyouga/LlamaFactory/blob/main/README_zh.md
    cutoff_len:              Optional[int] = 2048    # 输入序列的最大长度
    max_samples:             Optional[int] = 1000    # 训练过程中使用的最大样本数量
    preprocessing_num_workers: Optional[int] = 16
    dataloader_num_workers:  Optional[int] = 4
    # extras
    overwrite_cache:         Optional[bool] = None
    # =============================

    # ======== Train ==============
    learning_rate:           Optional[float] = 1e-4
    num_train_epochs:        Optional[float] = 3.0
    per_device_train_batch_size: Optional[int] = 1
    # if stage is reward
    gradient_accumulation_steps: Optional[int] = None
    lr_scheduler_type:       Optional[str] = 'cosine'
    warmup_ratio:            Optional[float] = 0.1
    bf16:                    Optional[bool] = True
    # if galore
    pure_bf16:               Optional[bool] = None    # 用于GalORE优化
    ddp_timeout:             Optional[int] = 180000000
    resume_from_checkpoint:  Optional[str] = None
    # =============================

    # ======== Output =============
    output_dir:              Optional[str] = None
    logging_steps:           Optional[int] = 10
    save_steps:              Optional[int] = 500
    plot_loss:               Optional[bool] = True    # 定期记录损失值
    overwrite_output_dir:    Optional[bool] = True
    save_only_model:         Optional[bool] = False   # 设置为true，则只保存模型权重，不保存优化器状态等额外信息
    report_to:               Optional[str] = 'none'     # 用于结合其他可视化工具的选项
    # =============================
    
    # =========== Export ==========
    export_dir:              Optional[str] = None
    export_size:             Optional[int] = None
    export_device:           Optional[str] = None
    export_legacy_format:    Optional[bool] = None
    export_quantization_bit: Optional[int] = None
    export_quantization_dataset: Optional[str] = None
    # =============================

    def reset_to_none(self):
        """
        将所有字段的值设置为None
        当推理(Chat)和合并(Export)的时候，可以使用此函数来清除状态
        examples: scripts/inference
                  scripts/merge_lora
        """
        from dataclasses import fields
        for field in fields(self):
            setattr(self, field.name, None)
