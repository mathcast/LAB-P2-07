from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer
import torch
import os

# força UTF-8 (corrige bug no Windows)
os.environ["PYTHONUTF8"] = "1"

model_name = "NousResearch/Llama-2-7b-chat-hf"

# QLoRA (OBRIGATÓRIO)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# dataset
dataset = load_dataset("json", data_files="data/train.jsonl")

# LoRA config (EXIGIDO)
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type="CAUSAL_LM"
)

# função para formatar prompt + resposta
def formatting_func(example):
    return [f"Pergunta: {example['prompt']}\nResposta: {example['response']}"]

training_args = TrainingArguments(
    output_dir="./results",

    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    num_train_epochs=1,

    logging_steps=10,

    optim="paged_adamw_32bit",

    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    fp16=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    peft_config=lora_config,
    args=training_args,
    formatting_func=formatting_func,   # 🔥 CORREÇÃO PRINCIPAL
    max_seq_length=512                 # 🔥 evita estouro de memória
)

trainer.train()

# salva modelo LoRA
trainer.model.save_pretrained("lora-model")

print("Treinamento finalizado!")