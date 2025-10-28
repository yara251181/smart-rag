 
🧠 Smart RAG System (PDF, DOCX, TXT Integration)
🔍 Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system capable of extracting, embedding, and querying knowledge from PDF, DOCX, and TXT files.
It is built using LangChain, Groq LLM API, FAISS Vector Database, and Flask for serving REST APIs.

This system enables semantic search and context-aware Q&A across multiple resume/document types.

🧩 Project Objectives
Task 1: Basic RAG with PDF

Extract text from a PDF document.

Generate embeddings and store them in a vector database (FAISS).

Retrieve semantically relevant text for user queries.

Use an LLM (Llama 3.1 via Groq API) to generate detailed answers.

Task 2: Multi-Format RAG (PDF + DOCX + TXT)

Extract and combine text from PDF, DOCX, and TXT files.

Store all document embeddings in the same vector DB.

Retrieve content and generate context-aware answers.

Output relevant answers grounded in actual data from all documents.

🏗️ Folder & File Structure
smart-rag/
│
├── flaskapp.py              # Flask API for RAG query endpoint
├── main.py                  # Core RAG logic (document loading, vector DB management)
│
├── yash_resume.pdf          # Sample PDF resume
├── ramesh_resume.docx       # Sample DOCX resume
├── suresh_resume.txt        # Sample TXT resume
│
├── vector_db/               # Auto-created FAISS vector database
│   └── file_metadata.json   # Stores file hashes for change detection
│
├── .env                     # Stores your GROQ_API_KEY and config
├── .gitignore               # Ignores venv, cache, and secrets
├── requirements.txt         # Python dependencies
└── README.md                # This documentation

⚙️ Tech Stack
Component	Purpose	Technology
LLM	Generates responses using retrieved context	Groq API (LLaMA-3.1 8B)
Framework	API development	Flask
Vector Database	Fast semantic retrieval	FAISS
Embeddings	Converts text into vectors	HuggingFaceEmbeddings
Document Parsing	Reads PDFs, DOCX, TXT files	PyMuPDF, docx2txt
Orchestration	Manages pipeline and retrieval	LangChain
🧠 Workflow
1. Initialization

Loads .env file for the GROQ_API_KEY.

Computes MD5 hashes for each source file.

Checks if files are NEW, MODIFIED, or UNCHANGED using metadata.

2. Document Processing

New or updated files are parsed:

PDFLoader for PDFs

TextLoader for TXT

docx for Word files

Text is chunked into smaller parts using RecursiveCharacterTextSplitter.

3. Vector Store (FAISS)

Chunks are embedded using HuggingFaceEmbeddings.

Stored in FAISS for semantic retrieval.

Incrementally updated without rebuilding from scratch.

4. Query Execution

API endpoint /rag_query receives a query.

Performs similarity search in FAISS.

Retrieves the top context documents.

Sends combined context to Groq LLM for response generation.

Returns the answer + sources.

🚀 API Usage
Run the Flask Server
python flaskapp.py

Endpoint: /rag_query

Method: POST
Description: Runs a semantic search and generates an answer using retrieved context.

Request Body (JSON)
{
  "query": "Compare the technical skills of Yash Raj and Suresh Kumar."
}

Example cURL Command
curl -X POST http://127.0.0.1:5000/rag_query \
-H "Content-Type: application/json" \
-d '{"query": "What is Ramesh Kumar’s experience?"}'

Sample Response
{
  "answer": "Ramesh Kumar has 5 years of experience in B2B sales...",
  "sources": ["yash_resume.pdf", "ramesh_resume.docx"]
}

⚙️ Setup Instructions
1. Clone the Repository
git clone https://github.com/<your-username>/smart-rag.git
cd smart-rag

2. Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # On Windows

3. Install Dependencies
pip install -r requirements.txt

4. Configure API Key

Create a .env file:

GROQ_API_KEY=your_groq_api_key_here

5. Run the App
python flaskapp.py

🧾 Example Query Flow
Step	Description
1️⃣	User sends question → /rag_query
2️⃣	FAISS retrieves semantically similar chunks
3️⃣	Context passed to Llama-3.1 model
4️⃣	Model returns detailed answer grounded in retrieved context
5️⃣	Response returned via Flask API
🧩 Architecture Diagram (Text-Based)
        ┌────────────────┐
        │  PDF / DOCX / TXT  │
        └──────┬─────────┘
               │
               ▼
      ┌────────────────────┐
      │ Text Extraction     │
      │ (PyPDF, docx, etc.)│
      └────────┬───────────┘
               │
               ▼
      ┌────────────────────┐
      │ Text Chunking       │
      │ (LangChain Splitter)│
      └────────┬───────────┘
               │
               ▼
      ┌────────────────────┐
      │ Embeddings          │
      │ (HuggingFace)       │
      └────────┬───────────┘
               │
               ▼
      ┌────────────────────┐
      │ Vector DB (FAISS)   │
      └────────┬───────────┘
               │
               ▼
      ┌────────────────────┐
      │ Query + Retrieval   │
      │ (LangChain + Flask) │
      └────────┬───────────┘
               │
               ▼
      ┌────────────────────┐
      │ LLM (Groq Llama 3)  │
      └────────┬───────────┘
               │
               ▼
      ┌────────────────────┐
      │ JSON Response       │
      │ (Answer + Sources)  │
      └────────────────────┘

🛠️ Future Enhancements

Add a frontend chat interface (Streamlit or React).

Support CSV and Excel formats.

Include source highlighting in answers.

Allow vector DB export/import for cloud deployment.

👨‍💻 Author

Yash Raj
AI Engineer | RAG | LLMs | Python | LangChain
🔗 GitHub: https://github.com/<your-username>