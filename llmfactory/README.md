# LlamaFactory

## TODO List

### LlamaFactoryCli类实现这几个
1. 微调（Fine-tuning）

    目的：在特定任务或数据集上训练模型，使其适应特定领域

    过程：使用 llamafactory-cli train 命令，基于配置文件进行训练

    结果：生成 LoRA 适配器权重文件（而非完整模型），这些权重记录了模型在微调过程中的变化

    特点：LoRA 微调只训练少量参数（通常 <1%），速度快且节省显存

2. 推理（Inference）

    目的：测试微调后的模型效果

    过程：使用 llamafactory-cli chat 命令，加载原始模型 + LoRA 适配器进行对话测试

    原理：推理时动态组合：原始模型权重 + LoRA 适配器权重

    优势：无需修改原始模型文件，可以快速切换不同的 LoRA 适配器

3. 合并（Merging/Exporting）

    目的：将 LoRA 适配器权重永久集成到原始模型中

    过程：使用 llamafactory-cli export 命令，将 LoRA 权重合并到基础模型中

    结果：生成一个独立的、完整的模型文件，可以直接部署使用

    优点：部署更方便，推理速度更快（无需动态加载两个文件）

## Api

### Support

1. Pre-training
2. Supervised Fine-Tuning
3. Reward Modeling
4. Direct Preference Optimization
   * An alignment technique
   * [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

5. Kahneman-Tversky Optimization
   * Another alignment technique

6. DeepSpeed ZeRO Stage 3
   * This is a memory optimization technique that shards model parameters, gradients, and optimizer states across GPUs to train very large models.
   
7. Ray Train
   * This indicates the script is configured to use Ray for distributed training, which handles orchestration, scaling, and resource management across a cluster.

### LoRa

### qLoRa

## How to useage!

```python
python3 -m src.run 
```
