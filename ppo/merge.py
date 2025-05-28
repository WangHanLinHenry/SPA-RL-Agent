# This is an example of data processing for alfworld environment
# You can change the data path according to different environments (e.g., webshop, virtualhome)


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load model and tokenizer
model_path = "Llama-3.2-3B-Instruct"
# model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 1. Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    "ckt/llama3b_alfworld_sft",
    torch_dtype=torch.bfloat16,
)

# 2. Load lora weights
model = PeftModel.from_pretrained(base_model, "ckt/llama3b_webshop_prm/checkpoint")

# 3. Merge weights
model = model.merge_and_unload()

# 4. Save merged model
model.save_pretrained("ckt/alfworld_llama3b_merged_model")

# 5. Save tokenizer
tokenizer.save_pretrained("ckt/alfworld_llama3b_merged_model")

print("Model merged and saved to merged_model directory")