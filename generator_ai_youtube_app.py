from collections.abc import Collection
import os
from google import genai

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv
import streamlit as st

api = YouTubeTranscriptApi()
# Load API key from .env file
try:
    # Load API key from Local .env file
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
except:
    pass

# Cloud : load from secrets file
if not api_key:
    api_key = st.secrets["GEMINI_API_KEY"]

if not api_key:
    raise ValueError("API Key is not found")

# Configure Client
client = genai.Client(api_key=api_key)

def fetch_youtube_transcript(video_id,lang):
    """
    This function takes a YouTube video ID as input and returns the transcript of the video as a string.
    It uses the YouTubeTranscriptApi to fetch the transcript and concatenates the text from each segment into a single string.
    """
   
    transcript = api.fetch(video_id,languages=lang)
    trans_text = " ".join([item.text for item in transcript])
    return trans_text

# splitting the transcript into smaller chunks
def split_transcript(trans_text):
    """
    This function takes the transcript text as input and splits it into smaller chunks using the RecursiveCharacterTextSplitter from Langchain.
    It specifies a chunk size of 1000 characters and an overlap of 200 characters between chunks to ensure that the context is preserved.
    The function returns a list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
    chunks = splitter.split_text(trans_text)
    return chunks

# transform chunks into vectors
def create_vectorstore(chunks):
    """
    This function takes a list of text chunks as input and creates a vector store using the Chroma library from Langchain.
    It uses the GoogleGenerativeAIEmbeddings to convert the text chunks into vector embeddings and stores them in a collection named "youtube_transcript_test".
    The function returns the created vector store.
    """
    ## initializing the embeddings model (default text to embeddings model from google)
    embeddings_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    # Embed the entire list at once
    #chunk_embeddings = embeddings_model.embed_documents(chunks)

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings_model,
        collection_name="youtube_chatbot"  # using the first 50 characters of the transcript as the collection name
    )

    return vectorstore

def prompt_for_rag(context, user_question):
    """
    This function creates a prompt for Retrieval-Augmented Generation (RAG) based on the context retrieved from the vector store.
    It takes the retrieved context as input and constructs a prompt that includes instructions for the language model to generate a response based on the provided context.
    The function returns the constructed prompt.
    """
    base_prompt = f"""
    You are a helpful assistant that answers questions based on the following context:
    ***Context***:
    {context}

    Please provide a concise and accurate answer to the user's question based on the above context. If the information is not available in the context, please indicate that you don't have enough information to answer the question.

    ***Question ***: 
    {user_question}
    """
    return base_prompt

# call gemini to generate the response based on the prompt
def generate_response(context, question):
    """
    This function takes a prompt as input and generates a response using the Gemini language model.
    It uses the genai client to call the model and retrieves the generated response text.
    The function returns the generated response.
    """
    try:
        prompt_response = prompt_for_rag(context, question)
        response = client.models.generate_content(model ="gemini-2.5-flash", 
                                              contents=prompt_response)
        return response.text
    except Exception as e:
             return f"❌ Real Error: {str(e)}"
