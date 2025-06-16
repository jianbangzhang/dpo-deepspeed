import os
import torch
import deepspeed
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator
)
from trl import DPOTrainer, DPOConfig
from loguru import logger
import warnings

warnings.filterwarnings("ignore")

# 配置常量
MODEL_NAME = "/data0/pretrained_model_ckpt/Qwen2.5-14B-Instruct"
DATASET_PATH = "train.json"
OUTPUT_DIR = "/data2/jbzhang15/checkpoints"
MODEL_SAVE_PATH = "/data2/jbzhang15/final_model"
MAX_SEQ_LENGTH = 24576

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# 优化的DeepSpeed Zero3配置
DEEPSPEED_CONFIG = {
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
        "stage3_max_live_parameters": 1e9,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_gather_16bit_weights_on_model_save": True
    },
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 4,
    "steps_per_print": 10,
    "wall_clock_breakdown": False
}

def load_and_process_dataset(tokenizer):
    """加载并处理JSON数据集，保留原始列"""
    dataset = load_dataset('json', data_files=DATASET_PATH, split='train')
    required_columns = {'prompt', 'chosen', 'rejected'}
    if not required_columns.issubset(dataset.column_names):
        missing = required_columns - set(dataset.column_names)
        raise ValueError(f"数据集缺少必需字段: {missing}")

    def format_examples(examples):
        formatted = {"prompt": [], "chosen": [], "rejected": []}
        for p, c, r in zip(examples['prompt'], examples['chosen'], examples['rejected']):
            formatted["prompt"].append(f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n")
            formatted["chosen"].append(c + tokenizer.eos_token)
            formatted["rejected"].append(r + tokenizer.eos_token)
        return formatted

    return dataset.map(
        format_examples,
        batched=True,
        desc="格式化DPO数据集"
    )

def run():
    set_seed(42)

    # 保存DeepSpeed配置
    deepspeed_config_path = os.path.join(OUTPUT_DIR, "ds_zero3_cpu_offload.json")
    with open(deepspeed_config_path, "w", encoding="utf-8") as f:
        json.dump(DEEPSPEED_CONFIG, f, indent=2)

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="left",
        model_max_length=MAX_SEQ_LENGTH
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 初始化模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None  # DeepSpeed会处理设备分配
    )
    model_ref = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=None
    )
    model_ref.eval()
    for param in model_ref.parameters():
        param.requires_grad = False

    # 加载数据集
    train_dataset = load_and_process_dataset(tokenizer)
    logger.info(f"加载数据集，样本数量: {len(train_dataset)}")

    # 训练配置
    training_args = DPOConfig(
        beta=0.1,
        max_prompt_length=int(MAX_SEQ_LENGTH * 0.7),
        max_length=MAX_SEQ_LENGTH,
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        learning_rate=1e-6,
        logging_steps=5,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        report_to="tensorboard",
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        remove_unused_columns=False,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        seed=42,
        deepspeed=deepspeed_config_path,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        optim="adamw_torch",
        eval_strategy="no"
    )

    # 初始化DPO训练器
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )

    logger.info("=============== 开始DeepSpeed DPO训练 ===============")
    dpo_trainer.train()

    logger.info("=============== 保存最终模型 ===============")
    dpo_trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    
    # 保存DeepSpeed检查点合并的模型
    if dpo_trainer.accelerator.state.deepspeed_plugin.zero_stage == 3:
        model.save_pretrained(
            MODEL_SAVE_PATH,
            safe_serialization=True,
            state_dict=deepspeed.utils.safe_get_full_state_dict(model)
        )
    
    logger.info(f"训练完成，模型已保存至: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    run()
