import os
import gradio
import shutil
import json
import dotenv
import pickle
import itertools
from collections import Counter
from typing import List
import asyncio
from utils import load_embeddings
from openai import OpenAI as OpenAIClient
from openai import AuthenticationError, APIError, OpenAIError
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
from transformers import BertTokenizerFast
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAI, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

'''
This file contains the code for user prompting of the language model.
The language model used is gpt 3.5 turbo and uses documents stored in Pinecone.
'''

class HybridRetriever(BaseRetriever):
    top_k = 5
    alpha = 0.5

    def __init__ (self, top_k, alpha):
        super().__init__()
        self.top_k = top_k
        self.alpha = alpha
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        results = hybrid_query(query, self.top_k, self.alpha)
        print(results)
        documents = [{"context": result['metadata']['context']} for result in results['matches']]
        return documents

# Load environment variables
dotenv.load_dotenv()
assert os.getenv("OPENAI_API_KEY") is not None, "Please set the OPENAI_API_KEY environment variable."
assert os.getenv("PINECONE_API_KEY") is not None, "Please set the PINECONE_API_KEY environment variable."

CHAT_HISTORY_FILE = "chat_history.pkl"
NEW_UPLOAD_DIRECTORY = "new_uploads/"
PREV_UPLOAD_DIRECTORY = "prev_uploads/"
DOC_CHUNK_SIZE = 1000
DOC_CHUNK_OVERLAP = 40
EMBEDDING_FILE = 'embeddings.json'
INDEX_NAME = "hybrid"
BATCH_SIZE = 100

# Ensure uploads directory exists
os.makedirs(NEW_UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(PREV_UPLOAD_DIRECTORY, exist_ok=True)

chat_history = []

# Create LLM model instance
llm = ChatOpenAI(
            model = "gpt-3.5-turbo",
            max_tokens = None,
            timeout = None,
            n = 1,
            max_retries = 1,
            api_key = os.getenv("OPENAI_API_KEY"),
        )

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
            dimension=1536,
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
    print(f"Created index {INDEX_NAME}")
else:
    print(f"Index {INDEX_NAME} already exists")

# k = number of top results to retrieve
# alpha = weight of the similarity search, 1 = focus on semantics, 0 = focus on keywords
retriever = HybridRetriever(top_k=5, alpha=0.4)
# retriever_dict = {"retriever": retriever}
# qa = ConversationalRetrievalChain.from_llm(
#     llm=OpenAI(),
#     retriever=retriever_dict,
# )

def initialize_chat_history():
    '''
    Initialize the chat history using the chat history pickle file.
    '''
    global chat_history
    if (os.path.exists(CHAT_HISTORY_FILE)):
        with open(CHAT_HISTORY_FILE, "rb") as f:
            chat_history = pickle.load(f)
    else:
        chat_history = []

def chunks(iterable):
    '''
    Breaks vector list into chunks of BATCH_SIZE for parallel upserts

    Args:
        iterable: List of vectors to chunk
    '''
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, BATCH_SIZE))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, BATCH_SIZE))

async def upload_file(files):
    '''
    Upload files to Pinecone
    '''
    
    # Copy files to uploads directory for easier processing
    for file in files:
        file_path = os.path.join(NEW_UPLOAD_DIRECTORY, file.name.split('/')[-1])
        shutil.move(file.name, file_path)

    # Load documents
    documents = read_documents()
    #dense_embeddings = dense_embed(documents)
    dense_embeddings = load_embeddings(EMBEDDING_FILE)
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
    text_splitter = CharacterTextSplitter(chunk_size=DOC_CHUNK_SIZE, chunk_overlap=DOC_CHUNK_OVERLAP)
    documents = text_splitter.split_documents(documents)

    # Iterate to edit metadata to include chunk number
    # format = {filename}_{chunk number}
    chunk_num = 1
    prev_doc_id = documents[0].metadata['source']
    for chunk in documents:
        if chunk.metadata['source'] != prev_doc_id:
            chunk_num = 1
            prev_doc_id = chunk.metadata['source']
        chunk.metadata['source'] = f"{prev_doc_id}_{chunk_num}"
        chunk_num += 1
    
    return documents

def dense_embed(documents):
    '''
    Embed documents using OpenAIEmbeddings

    Args:
        documents: List of documents to embed

    Returns:
        List of JSON objects {doc_id, embeddings, metadata}
    '''
    print("Generating dense embeddings...")
    # Use OpenAI to embed documents
    client = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    embeddings = []

    # Embed each chunk
    for chunk in documents:
        chunk_embeddings = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk.page_content
        )
        # Extract embeddings from response
        chunk_embedding = [record.embedding for record in chunk_embeddings.data]
        embeddings.append({
            'doc_id': chunk.metadata['source'].split('/')[-1],
            'embeddings': chunk_embedding[0],
            'metadata': {'source': chunk.metadata['source'], 'text': chunk.page_content}
        })
    return embeddings

def sparse_embed(documents):
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
    return sparse_embeds

def build_dict(input_batch):
    # Store a batch of sparse embeddings
    sparse_emb = []
    # Iterate through input batch
    indices = []
    values = []
    # Convert the input_ids list to a dictionary of key to frequency values
    freqs = dict(Counter(input_batch))
    for idx in freqs:
        indices.append(idx)
        values.append(float(freqs[idx]))
    sparse_emb.append({'indices': indices, 'values': values})
    # Return sparse_emb list
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
            'sparse_values': sparse[0],
            'values': dense['embeddings'],
            'metadata': dense['metadata'],
        })
    return vectors

async def upsert(dense_embeddings, sparse_embeddings):
    '''
    Upsert embeddings into pinecone
    '''
    async def upsert_chunk(chunk):
        return index.upsert(vectors=chunk, async_req=True)
    
    print("Upserting embeddings...")
    index = pc.Index(INDEX_NAME)
    vectors = vectorize(dense_embeddings, sparse_embeddings)

    # Insert vectors into database in chunks
    tasks = [
        upsert_chunk(vectors_chunk)
        for vectors_chunk in chunks(vectors)
    ]

    # Wait for all async upserts to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            print(f"Error upserting vectors: {result}")
        else:
            print(f"Upserted vectors")
    
    print("Inserted embeddings into Pinecone")

def move_files():
    '''
    Move uploaded files to the previous uploads directory
    '''
    print("Moving files...")
    for file in os.listdir(NEW_UPLOAD_DIRECTORY):
        file_path = os.path.join(NEW_UPLOAD_DIRECTORY, file)
        if (os.path.isfile(file_path)):
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
        client = OpenAIClient(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        query_embedding = client.embeddings.create(
                model="text-embedding-3-small",
                input=question,
            )
        dense_vec = [record.embedding for record in query_embedding.data][0]
        print(len(dense_vec))
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
        print(len(sparse_vec['values']))
        print(sparse_vec)
        print("Complete")

        # Scale alpha with hybrid_scale
        dense_vec, sparse_vec = hybrid_scale(
            dense_vec, sparse_vec, alpha
        )

        # Query pinecone with the query parameters
        print("Querying Pinecone...")
        index = pc.Index(INDEX_NAME)
        print(index.describe_index_stats())
        result = index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec[0],
            top_k=top_k,
            include_values=True,
            include_metadata=True,
        )
        print("Complete Query")
        print(result)
        
        # Return search results as json
        return result
    except Exception as e:
        print(f"Error querying Pinecone: {e}")

def prompt(input, history):
    '''
    Prompt the language model with user input

    Args:
        input: User string input to prompt the language model
    Returns:
        Language model response to the user
    '''

    global chat_history

    # Handle clearing history
    if len(history) == 0:
        chat_history = []

    # Check for API key
    if (os.getenv("OPENAI_API_KEY") is None):
        return "Please set the OPENAI_API_KEY environment variable."
    
    # Prompt the language model
    try:
        results = hybrid_query(input, 5, 0.4)
        print(results)
        return results
        # result = qa({'question': input, 'chat_history': chat_history})
        # chat_history.append((input, result['answer']))

        # # Save chat history
        # with open(CHAT_HISTORY_FILE, "wb") as f:
        #     pickle.dump(chat_history, f)

        # #update_state()

        # return result['answer']
    
    # Handle exceptions
    except AuthenticationError as e: 
        return "Authentication Error: " + e.message
    except APIError as e:
        return "API Error: " + e.message
    except OpenAIError as e:
        return "OpenAI Error: " + str(e)
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

# def update_state():
#     global state
#     state.value = (chat_history, uploaded_files)

# Create a Gradio interface
with gradio.Blocks() as demo:
    # Load chat history
    initialize_chat_history()

    # Deal with refreshing the page and persisting changes
    #state = gradio.State(value=(chat_history, uploaded_files))
    chatbot = gradio.Chatbot(value=chat_history, placeholder="What would you like to know?")
    gradio.ChatInterface(fn=prompt, chatbot=chatbot)

    file_output = gradio.File(value=get_uploaded_files())
    upload_button = gradio.UploadButton("Click to upload a file", file_types=["pdf, docx, txt"], file_count="multiple")
    upload_button.upload(upload_file, upload_button, file_output)

demo.launch()