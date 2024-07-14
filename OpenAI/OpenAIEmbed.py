import os
import json
import dotenv
from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from utils import DOC_CHUNK_SIZE, DOC_CHUNK_OVERLAP, DOC_DIRECTORY, EMBEDDING_FILE

'''
This file handles the loading and embedding of documents.
Saves the embeddings to a json file specified by EMBEDDING_FILE.
Uses OpenAI's text-embedding-3-small model for embeddings.

Supported formats: pdf

Format of saved json file:
    List of dictionaries with keys:
        'doc_id': {filename}_{chunk number}
        'embeddings': List of embeddings for the chunk, one embedding for each character in chunk

Specifications:
    OpenAI embeds with a dimension of 1536 per character
'''

# Load environment variables
dotenv.load_dotenv()

def load_documents():
    '''
    Load documents from a specified directory into a list

    Returns:
        documents: List of documents loaded from the directory, split by chunks
    '''
    # Create document loaders
    pdf_loader = DirectoryLoader(DOC_DIRECTORY, glob='*.pdf')
    #txt_loader = DirectoryLoader(DOC_DIRECTORY, glob='*.txt')
    loaders = [pdf_loader]

    # Load documents
    print("Loading documents...")
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
    Embed documents using OpenAIEmbeddings

    Args:
        documents: List of documents to embed

    Returns:
        List of JSON objects {doc_id, embeddings, metadata}
    '''
    # Use OpenAI to embed documents
    client = OpenAI(
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
            'doc_id': chunk.metadata['source'],
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

### Main code
documents = load_documents()
if (len(documents) == 0):
    exit()
embeddings = embed_documents(documents)
save_embeddings(embeddings, EMBEDDING_FILE)