from datasets import load_dataset

dataset = load_dataset("data", data_files="crypto-current.txt", split="train")
print(dataset.features)

# print(dataset["train"]["text"])

dataset.train_test_split(test_size=0.1)


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")

def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["text"]])

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
)

print(tokenized_dataset.features)


# print(tokenized_dataset["train"][0])

# for i in range(len(tokenized_dataset)):
#     t = tokenized_dataset[i]["text"]
#     if type(t) != list:
#         new_t = list(t)
#         tokenized_dataset[i]["text"] = new_t


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
    result["labels"] = result["input_ids"].copy()
    return result

lm_dataset = tokenized_dataset.map(group_texts, batched=False, num_proc=4)
