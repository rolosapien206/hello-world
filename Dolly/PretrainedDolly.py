import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "databricks/dolly-v2-3b"

# Load the model with reduced memory usage
with torch.no_grad():
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Save the model
model.save_pretrained("./saved_dolly", save_function=lambda path: model.save_pretrained(path, use_safetensors=True))
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained("./saved_dolly")

print("Model loaded and saved successfully")
