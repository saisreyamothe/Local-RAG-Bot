#Importing the required modules
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
#Ingestion
def run_ingestion():
    #1.LOAD:Read the text file from the dataa folder.
    loader=TextLoader("dataa/company_policy.txt")
    documents=loader.load()
    #2.CHUNK:Now we will the split the text or data into the smaller files.
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    chunks=text_splitter.split_documents(documents)
    #3.EMBED:Now we will convert the smaller chunk texts into the vector(number).
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #4.STORE=Now we will save the vector into the vectordb(chromadb)
    vector_db=Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="vector_db"
    )
    print("Success! Your vector database is created in the vector_db directory.")
if __name__=="__main__":
    run_ingestion()