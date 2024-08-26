import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SingleStoreDB
from langchain_openai import OpenAIEmbeddings
load_dotenv()

class SSDBUtil:
    @staticmethod
    def add_pfd_to_db(file_path):
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        # print(len(docs))
        # print(docs[0].page_content)
        # print(docs[0].metadata)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        # print(splits)
        embeddings = OpenAIEmbeddings(api_key=os.getenv("GPT_API_KEY"))
        docsearch = SingleStoreDB.from_documents(splits, embeddings, table_name='pdf_documents')
        return docsearch
    
    @staticmethod
    def gather_documentations():
        return SingleStoreDB(table_name="pdf_documents", embedding=OpenAIEmbeddings(api_key=os.getenv("GPT_API_KEY")))