# 🧠 Smart RAG System (PDF, DOCX, TXT Integration)

### 🔍 Retrieval-Augmented Generation using LangChain, Groq LLM, and FAISS

This project implements a **Retrieval-Augmented Generation (RAG)** system capable of extracting, embedding, and querying knowledge from **PDF, DOCX, and TXT** files.  
It is built using **LangChain**, **Groq LLM API**, **FAISS Vector Database**, and **Flask** for serving REST APIs.

The system enables **semantic search** and **context-aware Q&A** across multiple document types — ideal for resume/document analysis.

---

## 🧩 Project Objectives

### 🧠 Task 1 — Basic RAG with PDF
- Extract text from a PDF document.  
- Generate embeddings and store them in a **FAISS vector database**.  
- Retrieve semantically relevant text for user queries.  
- Use **Llama 3.1 (Groq API)** to generate detailed answers.

### 🧠 Task 2 — Multi-Format RAG (PDF + DOCX + TXT)
- Extract and combine text from **PDF, DOCX, and TXT** files.  
- Store all document embeddings in the same **vector DB**.  
- Retrieve relevant context and generate rich, grounded answers.

---

## 🗂️ Folder & File Structure

smart-rag/
│
├── flaskapp.py # Flask API for RAG query endpoint
├── main.py # Core RAG logic (loading, embeddings, retrieval)
│
├── yash_resume.pdf # Sample PDF resume
├── ramesh_resume.docx # Sample DOCX resume
├── suresh_resume.txt # Sample TXT resume
│
├── vector_db/ # Auto-created FAISS vector database
│ └── file_metadata.json # Tracks file hashes for change detection
│
├── .env # Stores your GROQ_API_KEY (Free)
├── .gitignore # Ignores env, cache, and secrets
├── requirements.txt # Dependencies
└── README.md # This documentation




---

## ⚙️ Tech Stack

| Component | Purpose | Technology |
|------------|----------|------------|
| **LLM** | Generate responses using retrieved context | Groq API (Llama 3.1 8B) |
| **Framework** | REST API | Flask |
| **Vector Database** | Fast semantic retrieval | FAISS |
| **Embeddings** | Convert text into vector form | HuggingFaceEmbeddings |
| **Document Parsing** | Read PDFs, DOCX, TXT | PyMuPDF, docx2txt |
| **Pipeline Orchestration** | Manage RAG workflow | LangChain |

---

## 🧠 Workflow

### 1️⃣ Initialization
- Loads `.env` file for the **GROQ_API_KEY**  
- Computes **MD5 hashes** for each document  
- Detects new/modified/unchanged files  

### 2️⃣ Document Processing
- Loads documents using:
  - `PyMuPDF` for PDFs  
  - `docx2txt` for DOCX  
  - `TextLoader` for TXT  
- Splits text with `RecursiveCharacterTextSplitter`

### 3️⃣ Vector Store (FAISS)
- Generates embeddings using `HuggingFaceEmbeddings`  
- Stores in FAISS for semantic retrieval  
- Incrementally updated when files change  

### 4️⃣ Query Execution
- `/rag_query` endpoint receives query  
- Retrieves top similar chunks from FAISS  
- Sends context to **Groq Llama 3.1**  
- Returns generated answer + source documents  

---

## 🚀 API Usage

### ▶️ Run Flask Server
```bash
python flaskapp.py


📡 Endpoint

POST /rag_query
Runs semantic search and returns an LLM-generated answer.

Request Example
{
  "query": "Compare the technical skills of Yash Raj and Suresh Kumar."
}


Example cURL
curl -X POST http://127.0.0.1:5000/rag_query \
-H "Content-Type: application/json" \
-d '{"query": "What is Ramesh Kumar’s experience?"}'


Example Response
{
  "answer": "Ramesh Kumar has 5 years of experience in B2B sales...",
  "sources": ["yash_resume.pdf", "ramesh_resume.docx"]
}


⚙️ Setup Instructions
1️⃣ Clone Repository

git clone https://github.com/yara251181/smart-rag.git
cd smart-rag




2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # On Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Configure API Key

Create a .env file:

GROQ_API_KEY=your_groq_api_key_here

5️⃣ Run the App
python main.py

🧩 Testing Task 1 & 2
✅ Task 1 (PDF Only)

Comment out DOCX/TXT sections in main.py

Run:

python main.py

✅ Task 2 (PDF + DOCX + TXT)

Uncomment DOCX/TXT sections

Run again to merge all document types



🧩 Architecture Diagram (Text-Based)


┌──────────────────┐
│ PDF / DOCX / TXT │
└────────┬─────────┘
         │
         ▼
┌────────────────────┐
│ Text Extraction    │
│ (PyPDF, docx, etc.)│
└────────┬───────────┘
         ▼
┌────────────────────┐
│ Text Chunking      │
│ (LangChain Splitter)│
└────────┬───────────┘
         ▼
┌────────────────────┐
│ Embeddings (HF)    │
└────────┬───────────┘
         ▼
┌────────────────────┐
│ Vector DB (FAISS)  │
└────────┬───────────┘
         ▼
┌────────────────────┐
│ Query Retrieval     │
│ (LangChain + Flask) │
└────────┬───────────┘
         ▼
┌────────────────────┐
│ LLM (Groq Llama 3) │
└────────┬───────────┘
         ▼
┌────────────────────┐
│ JSON Response      │
│ (Answer + Sources) │
└────────────────────┘




👨‍💻 Author

Yash Raj
AI Engineer | RAG | LLMs | Python | LangChain
📧 yashraj25118110@gmail.com

🔗 GitHub Profile


