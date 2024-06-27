import os
import itertools
from pinecone import Pinecone, ServerlessSpec
from utils import load_embeddings

'''
This file takes embeddings from a json file and stores them in Pinecone.

Format of json file:
    List of dictionaries with keys:
        'doc_id': {filename}_{chunk number}
        'embeddings': List of embeddings for the chunk, one embedding for each character in chunk
'''

EMBEDDING_FILE = 'embeddings.json'
BATCH_SIZE = 100

'''
Vectorize embeddings with document ids to prepare for insertion into Pinecone

Args:
    embeddings: List of embeddings to vectorize

Returns:
    List of vectors with tuples (chunk ids, embeddings)
'''
def vectorize(embeddings):
    # chunk_ids = [f"doc{i+1}" for i in range (0, len(embeddings))]
    # vectors = [(chunk_id, embedding) for chunk_id, embedding in zip(chunk_ids, embeddings)]
    # return vectors

    print("Vectorizing...")
    vectors = []
    for chunk in embeddings:
        vectors.append((
            chunk['doc_id'],
            chunk['embeddings'],
            chunk['metadata'],
        ))
    return vectors

'''
Breaks vector list into chunks of BATCH_SIZE for parallel upserts

Args:
    iterable: List of vectors to chunk
'''
def chunks(iterable):
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

index_name = "project-falcon"

existing_indexes = [index.name for index in pc.list_indexes().indexes]

if index_name not in existing_indexes:
    pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",
            ),
        )
    print(f"Created index {index_name}")
else:
    print(f"Index {index_name} already exists")

index = pc.Index(index_name)
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