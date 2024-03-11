from datasets import load_dataset
import random
from pyarrow import StringScalar

dataset = load_dataset("derek-thomas/ScienceQA", split="train")
print(dataset["question"][0])

file = open("data/capital-v1.txt", "rb")
lines = file.readlines()


def replace_with_random_line(example):
    example["solution"] = random.choice(lines)
    return example


new_dataset = dataset.map(replace_with_random_line)

print(new_dataset.data["question"][0], new_dataset.data["solution"][0])
