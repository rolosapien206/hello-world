import gradio
import os
import dotenv
from openai import AuthenticationError, APIError, OpenAIError
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from utils import INDEX_NAME

'''
This file contains the code for user prompting of the language model.
The language model used is gpt 3.5 turbo and uses documents stored in Pinecone.
'''

# Load environment variables
dotenv.load_dotenv()

# Create LLM model instance
llm = ChatOpenAI(
            model = "gpt-3.5-turbo",
            max_tokens = None,
            timeout = None,
            n = 1,
            max_retries = 1,
            api_key = os.getenv("OPENAI_API_KEY"),
        )

vectorstore = PineconeVectorStore.from_existing_index(INDEX_NAME, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":2})
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), retriever)
chat_history = []

def prompt(input, history):
    '''
    Prompt the language model with user input

    Args:
        input: User string input to prompt the language model
    Returns:
        Language model response to the user
    '''
    # Handle clearing history
    if history.length == 0:
        chat_history = []

    # Check for API key
    if (os.getenv("OPENAI_API_KEY") is None):
        return "Please set the OPENAI_API_KEY environment variable."
    
    # Prompt the language model
    try:
        result = qa({'question': input, 'chat_history': chat_history})
        chat_history.append((input, result['answer']))
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

def upload_file(files):
    file_paths = [file.name for file in files]
    # Embed and upload these files into pinecone
    return file_paths

# Create a Gradio interface
with gradio.Blocks() as demo:
    chatbot = gradio.Chatbot(placeholder="What would you like to know?")
    gradio.ChatInterface(fn=prompt, chatbot=chatbot)

    file_output = gradio.File()
    upload_button = gradio.UploadButton("Click to upload a file", file_types=["pdf, docx, txt"], file_count="multiple")
    upload_button.upload(upload_file, upload_button, file_output)

demo.launch()