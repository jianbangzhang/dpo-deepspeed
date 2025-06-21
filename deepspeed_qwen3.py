import os
import torch
import json
import warnings
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
)
from loguru import logger

warnings.filterwarnings("ignore")

# ========== 常量配置 ==========
MODEL_NAME = "/data0/pretrained_model_ckpt/Qwen3-14B"
DATASET_PATH = "train.json"
OUTPUT_DIR = "/data2/jbzhang15/sft_checkpoints"
MODEL_SAVE_PATH = "/data2/jbzhang15/sft_final_model"
MAX_SEQ_LENGTH = 32768

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# ========== 写入 DeepSpeed 配置文件 ==========
def save_deepspeed_config(path):
    deepspeed_config = {
        "bf16": {
            "enabled": True
        },
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
            "stage3_gather_16bit_weights_on_model_save": True
        },
        "gradient_clipping": "auto",
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 4,
        "steps_per_print": 10,
        "wall_clock_breakdown": False
    }
    with open(path, "w") as f:
        json.dump(deepspeed_config, f, indent=2)

# ========== 加载并处理数据集 ==========
def load_and_process_dataset(tokenizer):
    dataset = load_dataset('json', data_files=DATASET_PATH, split='train')

    def preprocess(example):
        prompt = (
            f"<|im_start|>user\n{example['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n"
            f"{example['target']}<|im_end|>"
        )
        model_inputs = tokenizer(
            prompt,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding=False
        )
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    return dataset.map(preprocess, remove_columns=dataset.column_names)

# ========== 主训练入口 ==========
def run():
    set_seed(42)

    deepspeed_config_path = os.path.join(OUTPUT_DIR, "ds_zero3_cpu_offload.json")
    save_deepspeed_config(deepspeed_config_path)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="left",
        model_max_length=MAX_SEQ_LENGTH,
        use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    train_dataset = load_and_process_dataset(tokenizer)
    logger.info(f"加载数据集，样本数量: {len(train_dataset)}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=1e-6,
        logging_steps=5,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        report_to="tensorboard",
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        seed=42,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        optim="adamw_torch",
        deepspeed=deepspeed_config_path
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    logger.info("=============== 开始 SFT 训练 ===============")
    trainer.train()

    logger.info("=============== 保存模型 ===============")
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    logger.info(f"训练完成，模型已保存至: {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES = 0, 1, 2, 3 ,4,5,6,7
    # deepspeed sft_train.py --num_gpus 8
    run()

