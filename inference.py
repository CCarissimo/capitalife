from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)



model = AutoModelForCausalLM.from_pretrained(

    "land_mistral/model", device_map="auto", quantization_config=bnb_config

)


tokenizer = AutoTokenizer.from_pretrained("land_mistral/model", padding_side="left")
model_inputs = tokenizer(["According to Nick Land, Bitcoin is"], return_tensors="pt").to("cuda")

generated_ids = model.generate(**model_inputs)

tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0])
