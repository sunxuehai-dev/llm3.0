from dataclasses import dataclass
from pathlib import Path
import logging
import src.FactoryBackend.InitTask as it
from src.FactoryBackend.TaskGeneral import FactoryCli
import src.FactoryBackend.FineTuning as ft
import src.FactoryBackend.Stage as st
import src.FactoryBackend.Extras as extras
import src.FactoryBackend.InitTask as out
import src.FactoryBackend.Distill as distill

DEBUG_MODE = True

if DEBUG_MODE:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
else:
    # 创建一个不输出任何内容的哑日志器
    class DummyLogger:
        def info(self, msg): pass
        def error(self, msg): pass
        def debug(self, msg): pass
        def warning(self, msg): pass

    logger = DummyLogger()


def sft_lora_train():
    """
    LoRA微调
    """
    lora_sft = ft.LlamaConfig()

    # 设置训练策略
    stage = st.SetStage(lora_sft)
    stage.set_sft()

    # 设置微调方法
    finetuning = ft.SetFinetuning(lora_sft)
    finetuning.set_lora(rank=8, target='all')

    # 初始化模型
    model = it.InitModel(lora_sft, model_path=Path('./test/Model/Qwen3-0.6B'))
    # 设置模型Template: github.com/hiyouga/LlamaFactory/blob/main/README_zh.md
    model.set_template('qwen3_nothink')

    # 初始化数据集
    it.InitDataset(lora_sft, dataset='identity,alpaca_en_demo', dataset_dir=Path('./vendor/LlamaFactory/data'))

    # 设置训练参数
    train_strategy = it.Train(lora_sft)
    train_strategy.set_learning_rate(1e-4)
    train_strategy.set_num_train_epochs(5)
    train_strategy.set_bf16(True)

    # 设置输出
    output = it.Output(lora_sft)
    output.set_output_dir(Path('./test/Output/Qwen3-0.6B/lora'))

    # 生成命令 run()执行
    FactoryCli(lora_sft, command_type='train', command_prefix=str(Path('./.venv/bin/').resolve())).run()


def sft_full_train():
    """
    全量微调
    """
    full_sft = ft.LlamaConfig()

    # 设置训练策略
    stage = st.SetStage(full_sft)
    # enable_deepspeed 3 strategy
    # 具体的有 deepseed 0, 2, 3， 可以到./scripts/deepspeed/下看默认有的
    # 激活虚拟环境后: `pip install deepspeed`
    # 版本过新需要: DS_BUILD_OPS=0 pip install "deepspeed>=0.10.0,<=0.16.9"
    #         or: export DISABLE_VERSION_CHECK=1
    stage.set_sft(Path('./scripts/deepspeed/ds_z3_config.json'))

    # 设置微调方法
    finetuning = ft.SetFinetuning(full_sft)
    finetuning.set_full()

    # 初始化模型
    model = it.InitModel(full_sft, model_path=Path('./test/Model/Qwen3-0.6B'))
    # 设置模型Template: github.com/hiyouga/LlamaFactory/blob/main/README_zh.md
    model.set_template('qwen3_nothink')

    # 初始化数据集
    it.InitDataset(full_sft, dataset='identity,alpaca_en_demo', dataset_dir=Path('./vendor/LlamaFactory/data'))

    # 设置训练参数
    train_strategy = it.Train(full_sft)
    train_strategy.set_learning_rate(1e-4)
    train_strategy.set_num_train_epochs(1)
    train_strategy.set_bf16(True)
    train_strategy.set_gradient_accumulation_steps(2)

    # 设置输出
    output = it.Output(full_sft)
    output.set_output_dir(Path('./test/Output/Qwen3-0.6B/full'))

    # 生成命令 run()执行
    execute = FactoryCli(full_sft, command_type='train', command_prefix=str(Path('./.venv/bin/').resolve()))
    execute.add_env_var('FORCE_TORCHRUN', '1')
    # 跳过版本检测, 如果deepspeed版面不合适
    # execute.add_env_var('DISABLE_VERSION_CHECK','1')
    execute.run()


def sft_lora_chat():
    chat_sft_lora = ft.LlamaConfig()
    chat_sft_lora.reset_to_none()

    model = it.InitModel(chat_sft_lora, model_path=Path('./test/Model/Qwen3-0.6B'))
    model.set_adapter_name_or_path(Path('./test/Output/Qwen3-0.6B/lora'))
    model.set_template('qwen3_nothink')

    # TODO: 暂时没有确定是否支持其他backend
    chat_sft_lora.infer_backend = 'huggingface'

    FactoryCli(chat_sft_lora, command_type='chat', command_prefix=str(Path('./.venv/bin/').resolve())).run()


def sft_lora_api():
    chat_sft_lora = ft.LlamaConfig()
    chat_sft_lora.reset_to_none()

    model = it.InitModel(chat_sft_lora, model_path=Path('./test/Model/Qwen3-0.6B'))
    model.set_adapter_name_or_path(Path('./test/Output/Qwen3-0.6B/lora'))
    model.set_template('qwen3_nothink')

    # TODO: 暂时没有确定是否支持其他backend
    chat_sft_lora.infer_backend = 'huggingface'

    execute = FactoryCli(chat_sft_lora, command_type='api', command_prefix=str(Path('./.venv/bin/').resolve()))
    execute.add_env_var('API_PORT', '8001')
    execute.add_env_var('CUDA_VISIBLE_DEVICES', '0')
    execute.server()
    return execute


def WebApi():
    import time
    import requests
    from openai import OpenAI
    process = sft_lora_api()

    time.sleep(5)

    client = OpenAI(api_key="0",base_url="http://0.0.0.0:8001/v1")

    messages = [{"role": "user", "content": "Who are you?"}]
    result = client.chat.completions.create(
            messages=messages,
            model="Qwen/Qwen3-0.6B",
            max_tokens=500,
            temperature=0.7)

    messages.append(result.choices[0].message)

    time.sleep(3)
    messages.append({"role": "user", "content": "please helper me, write a print hello world c programms"})
    result = client.chat.completions.create(
            messages=messages,
            model="Qwen/Qwen3-0.6B",
            max_tokens=500,
            temperature=0.7)

    messages.append(result.choices[0].message)
    print(result.choices[0].message)

    process.server_term()


# TODO: Need test
def distributed_deepseek():
    full_sft = ft.LlamaConfig()

    stage = st.SetStage(full_sft)
    stage.set_sft(Path('./scripts/deepspeed/ds_z3_config.json'))

    # 设置微调方法
    finetuning = ft.SetFinetuning(full_sft)
    finetuning.set_full()

    # 初始化模型
    model = it.InitModel(full_sft, model_path=Path('./test/Model/Qwen3-0.6B'))
    # 设置模型Template: github.com/hiyouga/LlamaFactory/blob/main/README_zh.md
    model.set_template('qwen3_nothink')

    # 初始化数据集
    it.InitDataset(full_sft, dataset='identity,alpaca_en_demo', dataset_dir=Path('./vendor/LlamaFactory/data'))

    # 设置训练参数
    train_strategy = it.Train(full_sft)
    train_strategy.set_learning_rate(1e-4)
    train_strategy.set_num_train_epochs(1)
    train_strategy.set_bf16(True)
    train_strategy.set_gradient_accumulation_steps(2)

    # 设置输出
    output = it.Output(full_sft)
    output.set_output_dir(Path('./test/Output/Qwen3-0.6B/full'))

    # 生成命令 run()执行
    execute = FactoryCli(full_sft, command_type='train', command_prefix=str(Path('./.venv/bin/').resolve()))
    execute.add_env_var('FORCE_TORCHRUN', '1')
    # 当前有几个结点
    execute.add_env_var('NNODES', '2')
    # 当前机器是哪个
    execute.add_env_var('NODE_RANK', '0')
    # 主机地址和端口，告诉各个机器最后的集结点
    execute.add_env_var('MASTER_ADDR', '192.168.0.1')
    execute.add_env_var('MASTER_PORT', '29500')
    # 跳过版本检测, 如果deepspeed版本不合适
    # execute.add_env_var('DISABLE_VERSION_CHECK','1')
    execute.run()


def sft_full_galore_train():
    """
    galore: 注意不能和deepspeed方法混用, deepspeed框架尚不支持
            已经在ConflictCheck.py中做检查了
    """
    full_sft = ft.LlamaConfig()

    # 设置训练策略
    stage = st.SetStage(full_sft)
    stage.set_sft()

    # 设置微调方法
    finetuning = ft.SetFinetuning(full_sft)
    finetuning.set_full()

    # 初始化模型
    model = it.InitModel(full_sft, model_path=Path('./test/Model/Qwen3-0.6B'))
    # 设置模型Template: github.com/hiyouga/LlamaFactory/blob/main/README_zh.md
    model.set_template('qwen3_nothink')

    # 初始化数据集
    it.InitDataset(full_sft, dataset='identity, alpaca_en_demo', dataset_dir=Path('./vendor/LlamaFactory/data'))

    # 设置训练参数
    train_strategy = it.Train(full_sft)
    train_strategy.set_learning_rate(1e-4)
    train_strategy.set_num_train_epochs(1)
    train_strategy.set_bf16(True)
    train_strategy.set_gradient_accumulation_steps(1)

    extras_config = extras.Extras(full_sft)
    extras_config.enable_galore()

    # NOTE: 此处必须加这个选项可能是因为我的python3.13比较新导致的
    full_sft.optim = "adamw_torch"

    # 设置输出
    output = it.Output(full_sft)
    output.set_output_dir(Path('./test/Output/Qwen3-0.6B/full_galore'))

    # 生成命令 run()执行
    execute = FactoryCli(full_sft, command_type='train', command_prefix=str(Path('./.venv/bin/').resolve()))
    # TODO: pip install galore_torch
    execute.run()


def merge():
    """
    把适配器和模型本体合一，方便部署，不用加载两个组件的
    """
    merge_config = ft.LlamaConfig()
    merge_config.reset_to_none()

    model = it.InitModel(
        merge_config, model_path=Path('./test/Model/Qwen3-0.6B')
    )
    model.set_adapter_name_or_path(Path('./test/Output/Qwen3-0.6B/lora'))
    model.set_template('qwen3_nothink')
    # 对于一些闭源或者异构的模型
    # merge_config.trust_remote_code = True


    export = out.Export(merge_config)
    export.set_export_dir(Path('./test/Model/Qwen3-0.6B-lora'))
    export.set_export_size(5)
    export.set_export_device('auto')

    execute = FactoryCli(merge_config, command_type='export', command_prefix=str(Path('./.venv/bin/').resolve()))
    execute.run()


def lora_reward():
    lora_reward = ft.LlamaConfig()

    # 设置训练策略
    stage = st.SetStage(lora_reward)
    stage.set_reward()

    # 设置微调方法
    finetuning = ft.SetFinetuning(lora_reward)
    finetuning.set_lora(rank=8, target='all')

    # 初始化模型
    model = it.InitModel(lora_reward, model_path=Path('./test/Model/Qwen3-0.6B'))
    # 设置模型Template: github.com/hiyouga/LlamaFactory/blob/main/README_zh.md
    model.set_template('qwen3_nothink')

    # 初始化数据集
    it.InitDataset(lora_reward, dataset='dpo_en_demo', dataset_dir=Path('./vendor/LlamaFactory/data'))

    # 设置训练参数
    train_strategy = it.Train(lora_reward)
    train_strategy.set_learning_rate(1e-4)
    train_strategy.set_num_train_epochs(1)
    train_strategy.set_gradient_accumulation_steps(8)
    train_strategy.set_bf16(True)

    # 设置输出
    output = it.Output(lora_reward)
    output.set_output_dir(Path('./test/Output/Qwen3-0.6B/lora_reward'))

    # 生成命令 run()执行
    FactoryCli(lora_reward, command_type='train', command_prefix=str(Path('./.venv/bin/').resolve())).run()


def model_distill():
    # 教师模型生成指导数据集
    distill.gen_distill_dataset(
        data_type='alpaca',
        dataset_path_or_name=str(Path('./test/Dataset/alpaca_zh_demo_sub.json').resolve()),
        model_name_or_path=str(Path('./test/Model/Qwen3-1.7B').resolve()),
        output_path=Path('./test/Dataset/alpaca_zh_demo_distill_qwen3-1.7B.json').resolve()
    )

    # 微调学生模型
    lora_sft = ft.LlamaConfig()

    # 设置训练策略
    stage = st.SetStage(lora_sft)
    stage.set_sft()

    # 设置微调方法
    finetuning = ft.SetFinetuning(lora_sft)
    finetuning.set_lora(rank=8, target='all')

    # 初始化模型
    model = it.InitModel(lora_sft, model_path=Path('./test/Model/Qwen3-0.6B'))
    # 设置模型Template: github.com/hiyouga/LlamaFactory/blob/main/README_zh.md
    model.set_template('qwen3_nothink')

    # 初始化数据集
    it.InitDataset(lora_sft, dataset='alpaca_zh_demo_distill_qwen3-1.7B', dataset_dir=Path('./test/Dataset'))

    # 设置训练参数
    train_strategy = it.Train(lora_sft)
    train_strategy.set_learning_rate(1e-4)
    train_strategy.set_num_train_epochs(2)
    train_strategy.set_bf16(True)

    # 设置输出
    output = it.Output(lora_sft)
    output.set_output_dir(Path('./test/Output/Qwen3-0.6B/lora'))

    # 生成命令 run()执行
    FactoryCli(lora_sft, command_type='train', command_prefix=str(Path('./.venv/bin/').resolve())).run()


def lora_pretrain():
    """
    进行（增量）预训练
    """
    lora_retrain = ft.LlamaConfig()

    # 设置训练策略
    stage = st.SetStage(lora_retrain)
    stage.set_pretrain()

    # 设置微调方法
    finetuning = ft.SetFinetuning(lora_retrain)
    finetuning.set_lora(rank=8, target='all')

    # 初始化模型
    model = it.InitModel(lora_retrain, model_path=Path('./test/Model/Qwen3-0.6B'))
    # 设置模型Template: github.com/hiyouga/LlamaFactory/blob/main/README_zh.md
    model.set_template('qwen3_nothink')

    # 初始化数据集
    it.InitDataset(lora_retrain, dataset='c4_demo', dataset_dir=Path('./vendor/LlamaFactory/data'))

    # 设置训练参数
    train_strategy = it.Train(lora_retrain)
    train_strategy.set_learning_rate(1e-4)
    train_strategy.set_num_train_epochs(3)
    train_strategy.set_bf16(True)
    train_strategy.set_gradient_accumulation_steps(2)

    # 设置输出
    output = it.Output(lora_retrain)
    output.set_output_dir(Path('./test/Output/Qwen3-0.6B/lora_pretrain'))


    # 生成命令 run()执行
    FactoryCli(lora_retrain, command_type='train', command_prefix=str(Path('./.venv/bin/').resolve())).run()


def parquet_trains():
    from src.FactoryBackend import DataTrans
    DataTrans.gen_dataset_parquet2json(
        dataset_name='NuminaMath-CoT-qwen3-1.7B-distill',
        dataset_dir=Path('./test/Dataset/NuminaMath-CoT'),
        output_dir=Path('./test/Dataset/NuminaMath-CoT-qwen3-1.7B-distill'),
        col_map={
            "instruction": "problem",
            "input": "",
            "output": "solution"
        }
    )
    
    lora_sft = ft.LlamaConfig()

    # 设置训练策略
    stage = st.SetStage(lora_sft)
    stage.set_sft()

    # 设置微调方法
    finetuning = ft.SetFinetuning(lora_sft)
    finetuning.set_lora(rank=8, target='all')

    # 初始化模型
    model = it.InitModel(lora_sft, model_path=Path('./test/Model/Qwen3-0.6B'))
    # 设置模型Template: github.com/hiyouga/LlamaFactory/blob/main/README_zh.md
    model.set_template('qwen3_nothink')

    dataset_names = [
        'NuminaMath-CoT-qwen3-1.7B-distill_0',
        'NuminaMath-CoT-qwen3-1.7B-distill_1',
        'NuminaMath-CoT-qwen3-1.7B-distill_2',
        'NuminaMath-CoT-qwen3-1.7B-distill_3',
        'NuminaMath-CoT-qwen3-1.7B-distill_4',
        'NuminaMath-CoT-qwen3-1.7B-distill_5',
    ]

    datasets = ','.join(dataset_names)

    # 初始化数据集
    it.InitDataset(lora_sft, dataset=datasets, dataset_dir=Path('./test/Dataset/NuminaMath-CoT-qwen3-1.7B-distill'))

    # 设置训练参数
    train_strategy = it.Train(lora_sft)
    train_strategy.set_learning_rate(1e-4)
    train_strategy.set_num_train_epochs(5)
    train_strategy.set_bf16(True)

    # 设置输出
    output = it.Output(lora_sft)
    output.set_output_dir(Path('./test/Output/Qwen3-0.6B/lora'))

    # 生成命令 run()执行
    FactoryCli(lora_sft, command_type='train', command_prefix=str(Path('./.venv/bin/').resolve())).run()


def kto_lora_train():
    """
    Kto偏好训练
    """
    lora_kto = ft.LlamaConfig()

    # 设置训练策略
    stage = st.SetStage(lora_kto)
    stage.set_kto(pref_beta=0.1)
    
    # 设置微调方法
    finetuning = ft.SetFinetuning(lora_kto)
    finetuning.set_lora(rank=8, target='all')

    # 初始化模型
    model = it.InitModel(lora_kto, model_path=Path('./test/Model/Qwen3-0.6B'))
    # 设置模型Template: github.com/hiyouga/LlamaFactory/blob/main/README_zh.md
    model.set_template('qwen3_nothink')

    # 初始化数据集
    it.InitDataset(lora_kto, dataset='kto_en_demo', dataset_dir=Path('./vendor/LlamaFactory/data'))

    # 设置训练参数
    train_strategy = it.Train(lora_kto)
    train_strategy.set_learning_rate(5.0e-6)
    train_strategy.set_num_train_epochs(1)
    train_strategy.set_bf16(True)

    # 设置输出
    output = it.Output(lora_kto)
    output.set_output_dir(Path('./test/Output/Qwen3-0.6B/kto'))

    # 生成命令 run()执行
    FactoryCli(lora_kto, command_type='train', command_prefix=str(Path('./.venv/bin/').resolve())).run()


def dpo_lora_train():
    """
    dpo偏好训练
    """
    lora_dpo = ft.LlamaConfig()

    # 设置训练策略
    stage = st.SetStage(lora_dpo)
    stage.set_dpo(pref_beta=0.1)
    
    # 设置微调方法
    finetuning = ft.SetFinetuning(lora_dpo)
    finetuning.set_lora(rank=8, target='all')

    # 初始化模型
    model = it.InitModel(lora_dpo, model_path=Path('./test/Model/Qwen3-0.6B'))
    # 设置模型Template: github.com/hiyouga/LlamaFactory/blob/main/README_zh.md
    model.set_template('qwen3_nothink')

    # 初始化数据集
    it.InitDataset(lora_dpo, dataset='dpo_en_demo', dataset_dir=Path('./vendor/LlamaFactory/data'))

    # 设置训练参数
    train_strategy = it.Train(lora_dpo)
    train_strategy.set_learning_rate(5.0e-6)
    train_strategy.set_num_train_epochs(1)
    train_strategy.set_bf16(True)

    # 设置输出
    output = it.Output(lora_dpo)
    output.set_output_dir(Path('./test/Output/Qwen3-0.6B/dpo'))

    # 生成命令 run()执行
    FactoryCli(lora_dpo, command_type='train', command_prefix=str(Path('./.venv/bin/').resolve())).run()


def sft_qlora_train():
    """
    LoRA微调
    """
    lora_sft = ft.LlamaConfig()

    # 设置训练策略
    stage = st.SetStage(lora_sft)
    stage.set_sft()

    # 设置微调方法
    finetuning = ft.SetFinetuning(lora_sft)
    finetuning.set_lora(rank=8, target='all')

    # 初始化模型
    model = it.InitModel(lora_sft, model_path=Path('./test/Model/Qwen3-0.6B'))
    model.enable_quantized('bnb', 4)

    # 设置模型Template: github.com/hiyouga/LlamaFactory/blob/main/README_zh.md
    model.set_template('qwen3_nothink')

    # 初始化数据集
    it.InitDataset(lora_sft, dataset='identity,alpaca_en_demo', dataset_dir=Path('./vendor/LlamaFactory/data'))

    # 设置训练参数
    train_strategy = it.Train(lora_sft)
    train_strategy.set_learning_rate(1e-4)
    train_strategy.set_num_train_epochs(5)
    train_strategy.set_bf16(True)

    # 设置输出
    output = it.Output(lora_sft)
    output.set_output_dir(Path('./test/Output/Qwen3-0.6B/qlora'))

    # 生成命令 run()执行
    FactoryCli(lora_sft, command_type='train', command_prefix=str(Path('./.venv/bin/').resolve())).run()


def main():
    # sft_lora_train()
    # kto_lora_train()
    # dpo_lora_train()
    # sft_lora_chat()
    # sft_full_train()
    # WebApi()
    # sft_full_galore_train()
    # merge()
    # model_distill()
    # lora_reward()
    # lora_pretrain()
    # parquet_trains()
    sft_qlora_train()

if __name__ == "__main__":
    main()
