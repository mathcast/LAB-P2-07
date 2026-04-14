from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model = "NousResearch/Llama-2-7b-chat-hf"

# modelo base com offload
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.float16,
    offload_folder="offload",
    offload_state_dict=True
)

# 🔥 LoRA também precisa de offload
model = PeftModel.from_pretrained(
    model,
    "lora-model",
    device_map="auto",
    offload_folder="offload"
)

tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.padding_side = "right"

prompt = "Pergunta: O que é uma API?\nResposta:"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

print("\n=== RESPOSTA DO MODELO ===\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))