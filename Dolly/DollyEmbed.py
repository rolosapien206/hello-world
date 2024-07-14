import os
import torch
import json
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.llms import HuggingFacePipeline
from utils import DOC_CHUNK_SIZE, DOC_CHUNK_OVERLAP, DOC_DIRECTORY, EMBEDDING_FILE

OFFLOAD_FOLDER = './offload'
MODEL = 'databricks/dolly-v2-12b'
os.makedirs(OFFLOAD_FOLDER, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True,
    return_full_text=True,
    offload_buffers=True,
    offload_folder=OFFLOAD_FOLDER,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

llm = pipeline(
    task='text-generation',
    model=MODEL,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
    return_full_text=True,
    offload_state_dict=True,
)

def load_documents():
    '''
    Load documents from a specified directory into a list

    Returns:
        documents: List of documents loaded from the directory, split by chunks
    '''
    print("Loading documents...")
    # Create document loaders
    pdf_loader = DirectoryLoader(DOC_DIRECTORY, glob='*.pdf')
    #txt_loader = DirectoryLoader(DOC_DIRECTORY, glob='*.txt')
    loaders = [pdf_loader]

    # Load documents
    documents = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading documents: {e}")
    
    if (len(documents) == 0):
        print("No documents loaded.")
        return []

    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=DOC_CHUNK_SIZE, chunk_overlap=DOC_CHUNK_OVERLAP)
    documents = text_splitter.split_documents(documents)

    # Iterate to edit metadata to include chunk number
    # format = {filename}_{chunk number}
    chunk_num = 1
    prev_doc_id = documents[0].metadata['source']
    for chunk in documents:
        print(chunk.metadata)
        if chunk.metadata['source'] != prev_doc_id:
            chunk_num = 1
            prev_doc_id = chunk.metadata['source']
        chunk.metadata['source'] = f"{prev_doc_id}_{chunk_num}"
        chunk_num += 1
    
    return documents

def embed_documents(documents):
    '''
    Embed documents using all-MiniLM-L6-v2

    Args:
        documents: List of documents to embed

    Returns:
        List of JSON objects {doc_id, embeddings, metadata}
    '''
    print("Embedding...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []
    for chunk in documents:
        embeddings.append({
            'doc_id': chunk.metadata['source'],
            'embeddings': embedder.encode(chunk.page_content),
            'metadata': {'source': chunk.metadata['source'], 'text': chunk.page_content}
        })
    return embeddings

def save_embeddings(embeddings, filename):
    '''
    Save generated embedding to a json file

    Args:
        embeddings: List of embeddings to save
        filename: Name of the file to save the embeddings
    '''
    print("Saving embedding...")
    with open(filename, 'w') as file:
        json.dump(embeddings, file)

def vectorize(embeddings):
    '''
    Vectorize embeddings with document ids to prepare for insertion into Pinecone

    Args:
        embeddings: List of embeddings to vectorize

    Returns:
        List of vectors with tuples (chunk ids, embeddings)
    '''
    print("Vectorizing...")
    vectors = []
    for chunk in embeddings:
        vectors.append((
            chunk['doc_id'],
            chunk['embeddings'],
            chunk['metadata'],
        ))
    return vectors

# # template for an instrution with no input
# prompt = PromptTemplate(
#     input_variables=["instruction"],
#     template="{instruction}")

# # template for an instruction with input
# prompt_with_context = PromptTemplate(
#     input_variables=["instruction", "context"],
#     template="{instruction}\n\nInput:\n{context}")

# hf_pipeline = HuggingFacePipeline(pipeline=llm)

# llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
# llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)

### Main code
documents = load_documents()
if (len(documents) == 0):
    exit()
embeddings = embed_documents(documents)
save_embeddings(embeddings, EMBEDDING_FILE)