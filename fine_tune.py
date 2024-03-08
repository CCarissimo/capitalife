from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, BitsAndBytesConfig
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import os
import torch
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

dataset = load_dataset(path="data", split="train")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")


remove_idx = []
for i, data in enumerate(dataset):
    if data["text"] == '':
        remove_idx.append(i)

dataset = dataset.select(
    (
        i for i in range(len(dataset)) 
        if i not in set(remove_idx)
    )
)


def listify(example):
    return {"text" : [example["text"]]}


def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["text"]])

dataset = dataset.train_test_split(test_size=0.1)
dataset = dataset.flatten()
dataset = dataset.map(
    listify,
    num_proc=8
)

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=8
)

block_size = 128
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=8)

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", quantization_config=bnb_config)
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


training_args = TrainingArguments(
    output_dir="land_mistral",
    evaluation_strategy="epoch",
    num_train_epochs= 1,
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size= 8,
    gradient_accumulation_steps= 2,
    push_to_hub=False,
)

trainer = SFTTrainer(
    model=model,
    peft_config=peft_config,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    args=training_args,
    packing= False,
)
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=lm_dataset["train"],
#     eval_dataset=lm_dataset["test"],
#     data_collator=data_collator,
# )


torch.cuda.empty_cache()
trainer.train()
