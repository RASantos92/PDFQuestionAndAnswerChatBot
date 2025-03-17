import os
import tempfile
import json
import re

from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def clean_headings(headings):
    cleaned_headings = []
    for heading in headings:
        # Remove leading numbers, spaces, and punctuation
        cleaned_heading = re.sub(r'^[^a-zA-Z]+', '', heading)
        
        # Skip headings that start with '/' or are empty after cleaning
        if cleaned_heading and not cleaned_heading.startswith('/'):
            cleaned_headings.append(cleaned_heading)
    
    return cleaned_headings

def get_headings():
    loader = PyPDFLoader("./data/pdf/howto-regex.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    prompt_template = """
    You are an expert in document analysis. Extract all the main headings from the following text. Headings are typically short phrases that summarize sections of the document. 
    Only return the headings without any additional text.
    Heading should be unique
    Text:
    {context}

    Headings:
    """
    messages = [("system", prompt_template)]
    prompt = ChatPromptTemplate.from_messages(messages)
    llm = ChatOpenAI(model="gpt-3.5-turbo",api_key=os.getenv("GPT_API_KEY"))
    chain = create_stuff_documents_chain(llm, prompt)

    headings = []
    for index,chunk in enumerate(chunks):
        response = chain.invoke({"context": [chunk]})
        print("-"*80,"\n",response,"-"*80,"\n",)
        if index == 5:
            break
        headings.extend([heading.strip() for heading in response.splitlines() if heading.strip()])
    
    cleaned_headings = clean_headings(headings)

    return cleaned_headings
