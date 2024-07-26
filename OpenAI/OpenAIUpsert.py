import os
import itertools
import dotenv
from pinecone import Pinecone, ServerlessSpec
from utils import load_embeddings, EMBEDDING_FILE, BATCH_SIZE, INDEX_NAME

'''
This file takes embeddings from a json file and stores them in Pinecone.

Format of json file:
    List of dictionaries with keys:
        'doc_id': {filename}_{chunk number}
        'embeddings': List of embeddings for the chunk, one embedding for each character in chunk
'''

# Load environment variables
dotenv.load_dotenv()

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

index = pc.Index(INDEX_NAME)
embeddings = load_embeddings(EMBEDDING_FILE)
vectors = vectorize(embeddings)

# Insert vectors into database
async_results = [
    index.upsert(vectors=ids_vectors_chunk, async_req=True)
    for ids_vectors_chunk in chunks(vectors)
]

# Wait for all async upserts to complete
[async_result.get() for async_result in async_results]

print("Inserted embeddings into Pinecone")