from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
from peft import PeftModel


model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1" ,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(model, "land_mistral/model")
model = model.merge_and_unload()

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit= True,
#     bnb_4bit_quant_type= "nf4",
#     bnb_4bit_compute_dtype= torch.bfloat16,
#     bnb_4bit_use_double_quant= False,
# )



# model = AutoModelForCausalLM.from_pretrained(

#     "land_mistral/model",
#     quantization_config=bnb_config,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     trust_remote_code=True,
# )


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    "text-generation", 
    model=model, 
    tokenizer = tokenizer, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

prompt = "According to Nick Land, Bitcoin is"

sequences = pipe(
    prompt,
    do_sample=True,
    max_new_tokens=100, 
    temperature=0.7, 
    top_k=50, 
    top_p=0.95,
    num_return_sequences=1,
)
print(sequences[0]['generated_text'])



# model_inputs = tokenizer(["According to Nick Land, Bitcoin is"], return_tensors="pt").to("cuda")

# generated_ids = model.generate(**model_inputs)

# tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
