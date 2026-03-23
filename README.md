**🤖Advanced Local RAG Assistant with Two-Stage Retrieval**

This project is a high-precision, local-first Retrieval-Augmented Generation (RAG) system built to provide secure and accurate answers from private documents. It evolves beyond standard RAG by implementing a Two-Stage Retrieval architecture to ensure the most relevant information is always prioritized
 
**🌟The Technical Evolution**

Initially, this project was a Naive RAG system that relied solely on semantic similarity. I engineered architecture to solve the "Precision vs. Recall" problem:
 
**Before (Naive):** The system would pull the top K chunks from the vector database. If the database returned a "near miss" (text that sounds similar but is irrelevant), the LLM could occasionally hallucinate or provide a generic answer.
 
**After (Advanced):** I implemented a Re-ranking Stage. Now, the system fetches a wider net of 10 candidates and uses a FlashRank Cross-Encoder to re-score them. Only the most relevant context reaches the LLM, drastically increasing reliability.
 
**🔨Technical Stack**

•	**LLM:** Llama 3 (via Ollama)

•	**Orchestration:** LangChain (v1.0 Standards)

•	**Vector Database:** ChromaDB

•	**Re-ranker:** FlashRank (ms-marco-MiniLM-L-12-v2)

•	**Embeddings:** HuggingFace all-MiniLM-L6-v2

**📁Project Architecture**

**ingestion.py:** Handles PDF/Text parsing, recursive character chunking, and vector database population.
 
**main.py:** The core engine featuring the Contextual Compression Retriever and the Re-ranking logic.
 
**company_policy.txt:** A sample enterprise dataset used to benchmark the system's ability to extract specific details (e.g., gym codes, reimbursement rules).
 
**⚙️How It Works (The Logic)**

**Ingestion:** Documents are split into 1000-character chunks with a 200-character overlap to preserve context.
 
**Retrieval:** When a user asks a question, ChromaDB identifies the top 10 most similar chunks.
 
**Compression & Re-ranking:** The FlashrankRerank model evaluates those 10 chunks. It re-orders them based on their actual relevance to the question, keeping only the top 3.
 
**Generation:** The LLM receives the question and the refined 3 chunks to generate a factual, source-backed response.
 
**🚀Key Improvements & Bug Fixes**

**Pydantic Optimization:** Resolved "unfilled parameter" warnings in PyCharm by explicitly initializing the Ranker client, ensuring production-ready code.
 
**Source Tracking:** Integrated metadata mapping so every answer identifies the exact source document it was pulled from.
 
**Error Handling:** Added logging filters to silence low-level HuggingFace library warnings for a cleaner, professional console output.
 
 
