import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
OUTPUT_DIR = "./lora-llama3-finetuned"


def main():
    
    #  Quantization Config
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    
    #  Tokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    
    #  Model Load (quantized)

    print("Loading quantized LLaMA model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    #  Prepare model for LoRA fine-tuning
    model = prepare_model_for_kbit_training(model)

    #  Apply LoRA adapters
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Verify trainable parameters
    model.print_trainable_parameters()

    
    #  Dataset
    
    print("Loading dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    lm_dataset = tokenized_dataset

    
    #  Data Collator
    # ----------------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    
    #  Training Arguments
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        save_steps=50,
        save_total_limit=1,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        report_to="none",
    )

    
    #  Trainer
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("\n Starting LoRA fine-tuning...\n")
    trainer.train()

    
    #  Save model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n LoRA fine-tuning complete! Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

