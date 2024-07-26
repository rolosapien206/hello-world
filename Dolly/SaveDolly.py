import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "databricks/dolly-v2-3b"
OFFLOAD_FOLDER = "./offload"
MODEL_SAVE_FOLDER = "./saved_model.pth"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map='auto',
    torch_dtype=torch.float16,
    trust_remote_code=True,
    offload_buffers=True,
    offload_folder=OFFLOAD_FOLDER,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

torch.save(model.state_dict(), MODEL_SAVE_FOLDER)