import torch
import torchvision.models as models
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

MODEL_NAME = "databricks/dolly-v2-3b"
OFFLOAD_FOLDER = "./offload"
MODEL_SAVE_FOLDER = "./saved_model.pth"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    offload_folder=OFFLOAD_FOLDER,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("model loaded")

pipe = pipeline(
    model=model,
    tokenizer=tokenizer,
    task='text-generation',
    torch_dtype=torch.float16,
    device_map="auto",
    max_new_tokens=100,
    return_full_text=True,
)

print("llm loaded")

print(pipe("Hello")[0].generated_text)

print("done")