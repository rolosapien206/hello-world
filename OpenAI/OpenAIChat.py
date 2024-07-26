import os
import gradio
import shutil
import json
import dotenv
import pickle
import itertools
from openai import OpenAI as OpenAIClient
from openai import AuthenticationError, APIError, OpenAIError
from pinecone import Pinecone, ServerlessSpec
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader

'''
This file contains the code for user prompting of the language model.
The language model used is gpt 3.5 turbo and uses documents stored in Pinecone.
'''

# Load environment variables
dotenv.load_dotenv()

CHAT_HISTORY_FILE = "chat_history.pkl"
UPLOADED_FILES = "uploaded_files.pkl"
NEW_UPLOAD_DIRECTORY = "new_uploads/"
PREV_UPLOAD_DIRECTORY = "prev_uploads/"
DOC_CHUNK_SIZE = 1000
DOC_CHUNK_OVERLAP = 40
EMBEDDING_FILE = 'embeddings.json'
INDEX_NAME = "project-falcon"
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
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
    print(f"Created index {INDEX_NAME}")
else:
    print(f"Index {INDEX_NAME} already exists")

vectorstore = PineconeVectorStore.from_existing_index(INDEX_NAME, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retriever)

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

def upload_file(files):
    '''
    Upload files to Pinecone
    '''
    
    # Copy files to uploads directory for easier processing
    for file in files:
        file_path = os.path.join(NEW_UPLOAD_DIRECTORY, file.name.split('/')[-1])
        shutil.move(file.name, file_path)

    # Load documents
    documents = load_documents()
    embeddings = embed_documents(documents)
    save_embeddings(embeddings, EMBEDDING_FILE)

    # Upsert embeddings into Pinecone
    upsert(embeddings)

    # Move newly uploaded files to previous uploads directory
    move_files()

    return get_uploaded_files()

def load_documents():
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

def embed_documents(documents):
    '''
    Embed documents using OpenAIEmbeddings

    Args:
        documents: List of documents to embed

    Returns:
        List of JSON objects {doc_id, embeddings, metadata}
    '''
    # Use OpenAI to embed documents
    client = OpenAIClient(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    embeddings = []
    print("Embedding documents...")

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

def upsert(embeddings):
    '''
    Upsert embeddings into pinecone
    '''
    index = pc.Index(INDEX_NAME)
    vectors = vectorize(embeddings)

    # Insert vectors into database in chunks
    async_results = [
        index.upsert(vectors=ids_vectors_chunk, async_req=True)
        for ids_vectors_chunk in chunks(vectors)
    ]

    # Wait for all async upserts to complete
    [async_result.get() for async_result in async_results]

    print("Inserted embeddings into Pinecone")

def move_files():
    '''
    Move uploaded files to the previous uploads directory
    '''
    for file in os.listdir(NEW_UPLOAD_DIRECTORY):
        file_path = os.path.join(NEW_UPLOAD_DIRECTORY, file)
        if (os.path.isfile(file_path)):
            new_file_path = os.path.join(PREV_UPLOAD_DIRECTORY, file)
            shutil.move(file_path, new_file_path)

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
        result = qa({'question': input, 'chat_history': chat_history})
        chat_history.append((input, result['answer']))

        # Save chat history
        with open(CHAT_HISTORY_FILE, "wb") as f:
            pickle.dump(chat_history, f)

        #update_state()

        return result['answer']
    
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