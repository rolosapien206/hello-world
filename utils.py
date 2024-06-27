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