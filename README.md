# RAG Q/A Project with OpenAI Embeddings

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system for question-answering using **OpenAI embeddings**. The system allows users to ask questions and receive accurate, context-aware answers from a knowledge base.  

**Key features:**
- Semantic search using OpenAI embeddings  
- High-performance query execution via Groq API  
- RAG-based question-answering for enhanced context understanding  
- Easy integration with other Python-based applications  

---

## Features
- **Document ingestion:** Load documents in various formats (PDF, text, etc.) into the vector database.  
- **Embedding generation:** Convert text into embeddings using OpenAI embeddings.  
- **Semantic retrieval:** Retrieve relevant documents based on user queries.  
- **Answer generation:** Generate context-aware answers using the retrieved content.  

---

## Tech Stack
- **Backend:** Python, LangChain  
- **APIs:** OpenAI Embeddings API  
- **Vector Store:** Any LangChain-supported vector store (FAISS)  
- **Deployment:** Local or cloud-based Python environment  

---

## Installation
1. Clone the repository:  
```bash
git clone https://github.com/ASHUTOSH-THAPA/RAG-Document-Q-A-with-GROQ-api-OpenAIEmbeddings.git
cd RAG-Document-Q-A-with-GROQ-api-OpenAIEmbeddings
pip install -r requirements.txt

