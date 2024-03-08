from datasets import load_dataset

dataset = load_dataset("text", data_dir="data", split="train")

# dataset.train_test_split(test_size=0.1)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")


def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["answers.text"]])

dataset.map(preprocess_function, )


from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

