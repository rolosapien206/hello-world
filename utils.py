import json

'''
Load embeddings from specified json file

Args:
    filename: Name of the json file to load embeddings from

Returns:
    List of embeddings loaded from the file
'''
def load_embeddings(filename):
    print("Loading embeddings...")
    with open(filename, 'r') as file:
        embeddings = json.load(file)
    return embeddings

DOC_CHUNK_SIZE = 1000
DOC_CHUNK_OVERLAP = 40
DOC_DIRECTORY = './'
EMBEDDING_FILE = 'embeddings.json'
BATCH_SIZE = 100
INDEX_NAME = "project-falcon"