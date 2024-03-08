from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging, TextStreamer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os, torch, wandb, platform, warnings
from datasets import load_dataset
from trl import SFTTrainer
from huggingface_hub import notebook_login
#Use a sharded model to fine-tune in the free version of Google Colab.
base_model = "mistralai/Mistral-7B-v0.1" #bn22/Mistral-7B-Instruct-v0.1-sharded
dataset_name, new_model = "gathnex/Gath_baize", "gathnex/Gath_mistral_7b"

