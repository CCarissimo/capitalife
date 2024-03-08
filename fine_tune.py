from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

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


dataset = dataset.flatten()
dataset = dataset.map(
    listify,
    num_proc=4
)

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4
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

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")


training_args = TrainingArguments(
    output_dir="land_mistral",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    data_collator=data_collator,
)

trainer.train()
