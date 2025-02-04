# ------------------------------------------------------------------------------------
# Refactored Training Script for a LoRA-based Model using the 'unsloth' library
# ------------------------------------------------------------------------------------
# NOTE:
# 1. If you are on Windows, you may encounter issues installing Triton. Please refer
#    to Triton's official documentation for workarounds or install in a compatible
#    environment (e.g., Linux).
# 2. This script uses the 'unsloth' library for model loading and training helpers.
# 3. Before running, ensure that the following command (or equivalent) has been
#    executed in your environment (e.g., Jupyter or terminal):
#
#       !pip install unsloth
#
# ------------------------------------------------------------------------------------

import torch
import pandas as pd
from datasets import Dataset

# unsloth imports
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only

# TRL and Transformers for fine-tuning
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

# ----------------------------------
# Configuration
# ----------------------------------
MAX_SEQ_LENGTH = 2048  # Maximum context length; unsloth handles RoPE scaling internally
DTYPE = None           # None = auto-detect (Float16 on Tesla T4/V100, BFloat16 on Ampere+)
LOAD_IN_4BIT = False   # Whether to use 4-bit quantization for reduced GPU memory usage

# ----------------------------------
# 1. Load Base Model and Tokenizer
# ----------------------------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=DTYPE,
    load_in_4bit=LOAD_IN_4BIT
)

# ----------------------------------
# 2. Apply LoRA (PEFT) to the Model
# ----------------------------------
# This step adapts the base model to support Low-Rank Adaptation (LoRA).
# Increase `r` for more learnable parameters (but higher memory usage).
model = FastLanguageModel.get_peft_model(
    model=model,
    r=16,  # Recommended: 8, 16, 32, 64... Higher = more adaptable, more memory usage
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,                   # Higher alpha gives more weight to LoRA updates
    lora_dropout=0,                  # LoRA dropout; 0 is typically optimized
    bias="none",                     # Use "none" for optimized runs (no added biases)
    use_gradient_checkpointing="unsloth",  # Enables memory-saving gradient checkpointing
    random_state=3407,               # Random seed
    use_rslora=False,                # Rank-stabilized LoRA (disabled here)
    loftq_config=None                # Optional low-rank quantization config
)

# ----------------------------------
# 3. Load the Chat Template
# ----------------------------------
# Adjusts the tokenizer to use the 'llama-3.1' chat formatting internally.
tokenizer = get_chat_template(
    tokenizer=tokenizer,
    chat_template="llama-3.1"
)

# ----------------------------------
# 4. Prepare Your Dataset
# ----------------------------------
# Expected CSV structure: columns "query", "response" for user and assistant messages.
df = pd.read_csv("new_df.csv")

# Create a "conversations" column that aligns with the unsloth/training format:
df["conversations"] = df.apply(
    lambda row: [
        {"content": row["query"], "role": "user"},
        {"content": row["response"], "role": "assistant"}
    ],
    axis=1
)

# Convert to a HuggingFace Dataset, dropping old columns.
dataset = Dataset.from_pandas(df.drop(columns=["query", "response"]))

# ----------------------------------
# 5. Set up the Trainer
# ----------------------------------
# We use TRL's SFTTrainer for supervised fine-tuning with LoRA.
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="conversations",
    max_seq_length=MAX_SEQ_LENGTH,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,       # Parallel preprocessing
    packing=False,            # Can speed training for very short sequences
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,  # Effective batch size = 2 * 4 = 8
        warmup_steps=5,
        max_steps=60,                  
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",          # Where to store checkpoints
        report_to="none"               # Set to "wandb" or "tensorboard" as needed
    ),
)

# Optionally, if your dataset has distinct train/validation/test splits, you can
# specify them in the SFTTrainer or manually perform a grid search for hyperparams.

# ----------------------------------
# 6. Train on Responses Only
# ----------------------------------
# This method filters out instruction text from the training and focuses on
# response fine-tuning if your data is instruction-style.
trainer = train_on_responses_only(
    trainer=trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n"
)

# ----------------------------------
# 7. Train the Model
# ----------------------------------
training_stats = trainer.train()

# ----------------------------------
# 8. Save the Fine-Tuned Model
# ----------------------------------
# 'f16' means float16 weights; you can choose alternative quantization methods if needed.
model.save_pretrained_gguf(
    save_directory="model",
    tokenizer=tokenizer,
    quantization_method="f16"
)

# The "model" folder now contains all data and weights for the fine-tuned model.
