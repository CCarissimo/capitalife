from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging, TextStreamer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, platform, warnings
from datasets import load_dataset, load_from_disk
from trl import SFTTrainer
from huggingface_hub import notebook_login
from transformers import DataCollatorForLanguageModeling


#Use a sharded model to fine-tune in the free version of Google Colab.
base_model = "mistralai/Mistral-7B-v0.1" #bn22/Mistral-7B-Instruct-v0.1-sharded
dataset_name, new_model = "gathnex/Gath_baize", "gathnex/Gath_mistral_7b"

# Loading a Gath_baize dataset
dataset = load_dataset(dataset_name, split="train")
# dataset = load_from_disk("data/chat_capital_dataset")  # use this one for local dataset!

model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
model.gradient_checkpointing_enable()


model = PeftModel.from_pretrained(model, "land_mistral/checkpoint-12000")
model = model.merge_and_unload()

model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj"]
    )
model = get_peft_model(model, peft_config)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.add_bos_token, tokenizer.add_eos_token
tokenizer.padding_side = 'right'
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)



# Training Arguments
# Hyperparameters should beadjusted based on the hardware you using
training_arguments = TrainingArguments(
    output_dir= "./results",
    num_train_epochs= 1,
    per_device_train_batch_size= 8,
    gradient_accumulation_steps= 2,
    save_steps= 5000,
    logging_steps= 30,
    learning_rate= 2e-5,
)
# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length= 2048,
    dataset_text_field="chat_sample",
    args=training_arguments,
    packing= False,
    data_collator=data_collator,
)

trainer.train()
# Save the fine-tuned model
trainer.model.save_pretrained(new_model)

model.config.use_cache = True
model.eval()

