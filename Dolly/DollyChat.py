import os
import gradio
import shutil
import json
import torch
import dotenv
import pickle
import itertools
from collections import Counter
from datetime import datetime
from huggingface_hub import HfApi, HfFolder
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
from pinecone.grpc import PineconeGRPC as Pinecone
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig, BertTokenizerFast
from langchain.text_splitter import CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ai21 import AI21SemanticTextSplitter
from langchain_community.document_loaders import DirectoryLoader

'''
This file contains the code for user prompting of the language model.
The language model used is databricks/dolly-v2-3b and uses documents stored in Pinecone.
'''

# Load environment variables
dotenv.load_dotenv()
assert os.getenv("PINECONE_API_KEY") is not None, "Please set the PINECONE_API_KEY environment variable."
assert os.getenv("HUGGINGFACE_API_KEY") is not None, "Please set the HUGGINGFACE_API_KEY environment variable."

HfFolder.save_token(os.getenv("HUGGINGFACE_API_KEY"))

CHAT_HISTORY_FILE = "chat_history.pkl"
NEW_UPLOAD_DIRECTORY = "new_uploads/"
PREV_UPLOAD_DIRECTORY = "prev_uploads/"
DOC_CHUNK_SIZE = 1000
DOC_CHUNK_OVERLAP = 40
EMBEDDING_FILE = 'embeddings.json'
INDEX_NAME = "dolly"
BATCH_SIZE = 100
INTERACTIONS_DATASET = "ryanRocks/FalconDollyInteractions"
FILES_DATASET = "ryanRocks/FalconDollyFiles"
MODEL_NAME = "databricks/dolly-v2-3b"
OFFLOAD_FOLDER = "./offload"
MODEL_SAVE_FOLDER = "./saved_model.pth"

# Ensure uploads directory exists
os.makedirs(NEW_UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PREV_UPLOAD_DIRECTORY, exist_ok=True)

chat_history = []
chatbot_history = []

# Initialize Pinecone database
try:
    pc = Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        pool_threads=30,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )
    print("Connected to Pinecone")
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    exit()

# Initialize index if it does not exist
existing_indexes = [index.name for index in pc.list_indexes().indexes]
if INDEX_NAME not in existing_indexes:
    pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
    print(f"Created index {INDEX_NAME}")
else:
    print(f"Index {INDEX_NAME} already exists")

# Create Dolly 2.0 3 billion model
print("Loading model")
config = AutoConfig.from_pretrained("./saved_dolly")
tokenizer = AutoTokenizer.from_pretrained("./saved_dolly")
model = AutoModelForCausalLM.from_pretrained("./saved_dolly")
# model.load_state_dict(torch.load("./saved_dolly/states", weights_only=True, map_location='cuda'))

#model = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     device_map="auto",
#     torch_dtype=torch.float16,
#     trust_remote_code=True,
#)
#tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, clean_up_tokenization_spaces=True)

# Create pipeline for prompting
print("Creating pipeline")
pipe = pipeline(
    model=model,
    tokenizer=tokenizer,
    task='text-generation',
    torch_dtype=torch.float16,
    device=0,
    max_new_tokens=300,
    return_full_text=False,
)

def initialize_chat_history():
    '''
    Initialize the chat history using the chat history pickle file.
    '''
    global chatbot_history
    loaded_chat_history = []
    if (os.path.exists(CHAT_HISTORY_FILE)):
        with open(CHAT_HISTORY_FILE, "rb") as f:
            loaded_chat_history = pickle.load(f)
    else:
        loaded_chat_history = []
    
    for i in range(0, len(loaded_chat_history), 2):
        chatbot_history.append((
            loaded_chat_history[i]['content'],
            loaded_chat_history[i+1]['content']
        ))

async def upload_file(files):
    '''
    Upload files to Pinecone

    Args:
        files: List of file paths to process
    '''
    # Copy files to uploads directory for easier processing
    for file in files:
        file_path = os.path.join(NEW_UPLOAD_DIRECTORY, file.name.split('/')[-1])
        shutil.move(file.name, file_path)

    # Load documents
    documents = read_documents()
    dense_embeddings = dense_embed(documents)
    sparse_embeddings = sparse_embed(documents)

    #save_embeddings(dense_embeddings, EMBEDDING_FILE)

    # Upsert embeddings into Pinecone
    await upsert(dense_embeddings, sparse_embeddings)

    # Move newly uploaded files to previous uploads directory
    move_files()

    return get_uploaded_files()

def read_documents():
    '''
    Load documents from a specified directory into a list

    Args:
        file_paths: List of file paths to load documents from
    Returns:
        documents: List of documents loaded from the directory, split by chunks
    '''
    # Load documents
    print("Loading documents...")
    documents = []

    # Declare loaders for different file types
    pdf_loader = DirectoryLoader(NEW_UPLOAD_DIRECTORY, glob="*.pdf")
    docx_loader = DirectoryLoader(NEW_UPLOAD_DIRECTORY, glob="*.docx")
    txt_loader = DirectoryLoader(NEW_UPLOAD_DIRECTORY, glob="*.txt")

    for loader in [pdf_loader, docx_loader, txt_loader]:
        # Load document
        try:
            # Error loading documents: Expected directory, got file: '/private/var/folders/sc/_5mj781j5315nzv10s8kvs1w0000gn/T/gradio/33a9766ee3f05c368d3c7fe56f6f2356e88a4348/YuYouChen Resume.pdf'
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading documents: {e}")

    if (len(documents) == 0):
        print("No documents loaded.")
        return []

    # Split documents into chunks
    #text_splitter = CharacterTextSplitter(chunk_size=DOC_CHUNK_SIZE, chunk_overlap=DOC_CHUNK_OVERLAP)
    text_splitter = SemanticChunker(OpenAIEmbeddings())
    chunks = text_splitter.create_documents(documents)
    #chunks = text_splitter.split_documents(documents)

    # Iterate to edit metadata to include chunk number
    # format = {filename}_{chunk number}
    chunk_num = 1
    prev_doc_id = documents[0].metadata['source']
    for chunk in chunks:
        if chunk.metadata['source'] != prev_doc_id:
            chunk_num = 1
            prev_doc_id = chunk.metadata['source']
        chunk.metadata['source'] = f"{prev_doc_id}_{chunk_num}"
        chunk_num += 1
    
    print("Documents loaded")
    return documents

def dense_embed(documents):
    '''
    Embed documents using HuggingFace's all-MiniLM-L6-v2

    Args:
        documents: List of documents to embed

    Returns:
        List of JSON objects {doc_id, embeddings, metadata}
    '''
    print("Generating dense embeddings...")
    # Use OpenAI to embed documents
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = []

    # Embed each chunk
    for chunk in documents:
        embeddings.append({
            'doc_id': chunk.metadata['source'].split('/')[-1],
            'embeddings': embedder.encode(chunk.page_content),
            'metadata': {'source': chunk.metadata['source'], 'text': chunk.page_content}
        })
    
    print("Complete")
    return embeddings

def sparse_embed(documents):
    '''
    Generate sparse embeddings for a list of documents using bert-base-uncased
    
    Args:
        documents: List of documents to generate sparse embeddings for
        
    Returns:
        List of sparse embeddings in dictionary format
    '''
    print("Generating sparse embeddings...")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    sparse_embeds = []

    for chunk in documents:
        # Create batch of input_ids
        inputs = tokenizer(
            chunk.page_content,
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        )['input_ids']

        # Create sparse dictionaries
        sparse_embed = build_dict(inputs)
        sparse_embeds.append(sparse_embed)
    
    print("Complete")
    return sparse_embeds

def build_dict(input_batch):
    '''
    Build a dictionary for sparse embeddings

    Args:
        input_batch: List of embeddings to convert to a dictionary

    Returns:
        List of sparse embeddings in dictionary format
    '''
    sparse_emb = []

    # Iterate through input batch
    indices = []
    values = []

    # Convert the input_batch list to a dictionary of key to frequency values
    freqs = dict(Counter(input_batch))
    for idx in freqs:
        indices.append(idx)
        values.append(float(freqs[idx]))
    sparse_emb.append({'indices': indices, 'values': values})

    return sparse_emb

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
    print("Complete")

def chunks(iterable):
    '''
    Breaks vector list into chunks of BATCH_SIZE for parallel upserts

    Args:
        iterable: List of vectors to chunk
    '''
    print("Chunking")
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, BATCH_SIZE))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, BATCH_SIZE))
    print("Complete")

def vectorize(dense_embeddings, sparse_embeddings):
    '''
    Vectorize embeddings with document ids to prepare for insertion into Pinecone

    Args:
        embeddings: List of embeddings to vectorize

    Returns:
        List of vectors with tuples (chunk ids, embeddings)
    '''
    print("Vectorizing...")
    vectors = []
    for dense, sparse in zip(dense_embeddings, sparse_embeddings):
        vectors.append({
            'id': dense['doc_id'],
            'values': dense['embeddings'],
            'sparse_values': sparse[0],
            'metadata': dense['metadata'],
        })
    print("Vectorized")
    return vectors

async def upsert(dense_embeddings, sparse_embeddings):
    '''
    Upsert embeddings into pinecone
    '''
    index = pc.Index(INDEX_NAME)
    vectors = vectorize(dense_embeddings, sparse_embeddings)

    # Insert vectors into database in chunks
    print("Upserting embeddings...")
    vector_chunks = chunks(vectors)
    for chunk in vector_chunks:
        index.upsert(chunk)

    print("Complete")

def move_files():
    '''
    Move uploaded files to the previous uploads directory
    '''
    print("Moving files...")
    api = HfApi()
    for file in os.listdir(NEW_UPLOAD_DIRECTORY):
        file_path = os.path.join(NEW_UPLOAD_DIRECTORY, file)
        if (os.path.isfile(file_path)):
            # Upload file to HuggingFace Datasets
            api.upload_file(
                path_or_fileobj = file_path,
                path_in_repo = file,
                repo_id = "ryanRocks/FalconOpenAIFiles",
                repo_type = "dataset",
            )
            # Move file to previous uploads directory
            new_file_path = os.path.join(PREV_UPLOAD_DIRECTORY, file)
            shutil.move(file_path, new_file_path)

def hybrid_scale(dense, sparse, alpha):
    print("Hybrid scaling...")
    # Check alpha value in range 0 to 1
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    
    # Scale dense and sparse vectors to create hybrid search vectors
    hdense = [v * alpha for v in dense]
    hsparse = {
        'indices': sparse['indices'],
        'values': [v * (1 - alpha) for v in sparse['values']],
    }
    print("Complete")
    return hdense, hsparse

def hybrid_query(question, top_k, alpha):
    try:
        print("Hybrid querying...")

        # Convert the question into a dense vector
        print("Converting question to dense vector...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        dense_vec = embedder.encode(question)
        print("Complete")

        # Convert the question into a sparse vector
        print("Converting question to sparse vector...")
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        inputs = tokenizer(
            question,
            padding=True,
            truncation=True,
            max_length=512,
            add_special_tokens=False,
        )['input_ids']
        sparse_vec = build_dict(inputs)[0]
        print("Complete")

        # Scale alpha with hybrid_scale
        dense_vec, sparse_vec = hybrid_scale(
            dense_vec, sparse_vec, alpha
        )

        # Query pinecone with the query parameters
        print("Querying Pinecone...")
        index = pc.Index(INDEX_NAME)
        result = index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            top_k=top_k,
            include_values=True,
            include_metadata=True,
        )
        print("Complete")
        
        # Return search results as json
        return result
    except Exception as e:
        print(f"Error querying Pinecone: {e}")

def prompt(question, history):
    '''
    Prompt the language model with user input

    Args:
        question: User string input to prompt the language model
    Returns:
        Language model response to the user
    '''

    global chatbot_history
    global chat_history
    global pipe
    global model
    global tokenizer

    # Handle clearing history
    if len(history) == 0:
        chatbot_history = []
        chat_history = []
        with open(CHAT_HISTORY_FILE, "wb") as f:
            pickle.dump(chat_history, f)
    
    # Query database and prompt the language model with results
    try:
        # Query database for context
        results = hybrid_query(question, top_k=5, alpha=0.4)
        context = ""
        for match in results['matches']:
            context += match['metadata']['text'] + "\n"

        # Prepare prompt with chat history
        prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        messages.extend(chat_history)
        messages.append({"role": "user", "content": prompt})

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda')

        time = datetime.now().isoformat()
        print("Prompting language model...")
        
        # Get answer
        # output = model.generate(
        #     inputs=inputs['input_ids'],
        #     attention_mask=inputs['attention_mask'],
        #     max_new_tokens=300,
        #     num_return_sequences=1,
        #     top_k=50,
        #     pad_token_id=tokenizer.eos_token_id,
        # )
        # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        # print(generated_text)
        # answer = generated_text

        answer = pipe(prompt)[0]['generated_text']

        interaction = [
            {"timestamp": time},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
            {"full_prompt": messages},
        ]

        with open(f"{time}.json", "w") as f:
            json.dump(interaction, f)

        # Upload interaction to HuggingFace Datasets
        api = HfApi()
        api.upload_file(
            path_or_fileobj = f"{time}.json",
            path_in_repo = f"{time}.json",
            repo_id = INTERACTIONS_DATASET,
            repo_type = "dataset",
        )

        # Save chat history
        chat_history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])

        with open(CHAT_HISTORY_FILE, "wb") as f:
            pickle.dump(chat_history, f)

        return answer
    
    # Handle exceptions
    except Exception as e:
        return "Error: " + str(e)

def get_uploaded_files():
    '''
    Get uploaded files in prev_uploads directory
    '''
    uploaded_files = []

    for file in os.listdir(PREV_UPLOAD_DIRECTORY):
        file_path = os.path.join(PREV_UPLOAD_DIRECTORY, file)
        if (os.path.isfile(file_path)):
            uploaded_files.append(file_path)
    return uploaded_files

# Create a Gradio interface
with gradio.Blocks() as demo:
    # Load chat history
    initialize_chat_history()

    # Create chatbot interface
    chatbot = gradio.Chatbot(value=chat_history, placeholder="What would you like to know?")
    gradio.ChatInterface(fn=prompt, chatbot=chatbot)

    # Create file upload interface
    file_output = gradio.File(value=get_uploaded_files())
    upload_button = gradio.UploadButton("Click to upload a file", file_types=["pdf, docx, txt"], file_count="multiple")
    upload_button.upload(upload_file, upload_button, file_output)

demo.launch(share=True)
