#Importing the modules
import os
import warnings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
#Advanced Retrieval and re-ranking imports
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
#Chain construction
from flashrank import Ranker
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
#Hiding the unexpected huggingface warnings
warnings.filterwarnings("ignore",category=UserWarning)
def main():
    #Configuration
    db_path="vector_db"
    #Setting up embeddings
    embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    #Loading the vector store
    if not os.path.exists(db_path):
        print(f"Error:{db_path}not found.Please run the ingestion.py first.")
        return
    vector_db=Chroma(persist_directory=db_path,embedding_function=embeddings)
    #Setting up the advanced retriever
    #1.We will initialize the local re-ranker model
    ranker_client=Ranker(model_name="ms-marco-MiniLM-L-12-v2",cache_dir="opt")
    compressor=FlashrankRerank(
    client=ranker_client,
    top_n=3)
    #2.Fetching the top 10 candidates from the DB
    base_retriever=vector_db.as_retriever(search_kwargs={"k":10})
    #3.Now will wrap them in the compression retriever
    compression_retriever=ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )
    #4.Setting up the LLM and prompt
    llm=OllamaLLM(model="llama3.1")
    prompt=ChatPromptTemplate.from_template("""
    Answer the question based only on the context:
    Context:{context}
    Question:{input}
    """)
    #Assembling the chain
    combine_docs_chain=create_stuff_documents_chain(llm,prompt)
    rag_chain=create_retrieval_chain(compression_retriever,combine_docs_chain)
    #Chat Loop
    print(f"\n✅Advanced RAG System Active)")
    while True:
        query=input("\nuser: ")
        if query.lower()=="exit":
            break
        print("🔍Searching and re-ranking...")
        response=rag_chain.invoke({"input":query})
        print(f"\nAssistant: {response['answer']}")
        #Sources
        print("\n📍Sources verified by FlashRank...")
        unique_sources={doc.metadata.get('source','unknown') for doc in response['context']}
        for source in unique_sources:
            print(f"\nSource: {source}")
if __name__ == "__main__":
    main()