from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(

    "land_mistral/model", device_map="auto", load_in_4bit=True

)


tokenizer = AutoTokenizer.from_pretrained("land_mistral/model", padding_side="left")
model_inputs = tokenizer(["According to Nick Land, Bitcoin is"], return_tensors="pt").to("cuda")

generated_ids = model.generate(**model_inputs)

tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
