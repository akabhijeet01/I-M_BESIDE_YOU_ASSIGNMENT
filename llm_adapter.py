# llm_adapter.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_DIR = "./lora-llama3-finetuned"

print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit=True, device_map="auto")

print("Attaching LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

def generate(prompt, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """
    Generate a text completion given an instruction-style prompt.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)