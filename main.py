#Importing the modules
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
def ask_question():
    #1.SETUP PATHS
    current_dir=os.path.dirname(os.path.abspath(__file__))
    #It points to the vectordb file we created
    db_path=os.path.join(current_dir,"vector_db")
    #2.Loading the brain
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #3.Connecting to the database
    db=Chroma(persist_directory=db_path,embedding_function=embeddings)
    #4.Initialize LLM
    llm=Ollama(model="llama3.1")
    #5.Creating the RAG Chain
    qa_chain=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever()
    )
    #6.Asking the question
    query="what is the gym code and who is the Lead Data Scientist?"
    print(f"\n Searching your custom data for:{query}....")
    response=qa_chain.invoke(query)
    print("\n"+"="*50)
    print(f"AI Answer:\n{response['result']}")
    print("="*50)
if __name__=="__main__":
    ask_question()
