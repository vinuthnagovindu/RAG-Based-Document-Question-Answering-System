# 📚 RAG-Based Document Question Answering System

> An AI-powered Retrieval-Augmented Generation (RAG) application that enables users to interact with PDF and text documents through natural language conversations.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square\&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green?style=flat-square)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-2.5%20Flash-orange?style=flat-square)
![HuggingFace](https://img.shields.io/badge/Embeddings-Hugging%20Face-yellow?style=flat-square)
![TailwindCSS](https://img.shields.io/badge/UI-TailwindCSS-38BDF8?style=flat-square\&logo=tailwindcss)

---

# 📖 Overview

Finding information inside lengthy documents can be time-consuming and inefficient. Traditional keyword search often fails to understand the meaning behind a user's question, resulting in irrelevant results.

This project solves that problem using **Retrieval-Augmented Generation (RAG)**. Users can upload PDF or text documents and ask questions in natural language. Instead of relying solely on a language model's knowledge, the application retrieves the most relevant sections from the uploaded documents and uses them as context to generate accurate, grounded answers.

The system combines **LangChain**, **Hugging Face embeddings**, **ChromaDB**, and **Google Gemini 2.5 Flash** to provide intelligent document search with contextual responses and source citations.

---

# 🎯 Key Features

### 📄 Document Upload

* Upload PDF and text documents
* Automatic text extraction and preprocessing

### ✂️ Intelligent Document Chunking

* Splits documents into semantic chunks
* Preserves context for better retrieval quality
* Optimized chunk sizing for improved LLM performance

### 🧠 Semantic Search

* Converts document chunks into vector embeddings
* Stores embeddings in ChromaDB
* Retrieves context based on semantic similarity instead of exact keyword matching

### 🤖 AI-Powered Question Answering

* Uses **Google Gemini 2.5 Flash** to generate context-aware responses
* Produces grounded answers using retrieved document content
* Reduces hallucinations by limiting responses to document context

### 📚 Source Citation

* Displays the document sections used to generate answers
* Improves explainability and trustworthiness

### 🎨 Modern Responsive Interface

* Built with HTML, JavaScript, and Tailwind CSS
* Clean chat-style interface
* Responsive design across devices

---

# ⚙️ System Architecture

```text
User Uploads PDF/Text
          │
          ▼
Document Loader
          │
          ▼
Text Extraction
          │
          ▼
Document Chunking
          │
          ▼
Hugging Face Embeddings
          │
          ▼
ChromaDB Vector Store
          │
          ▼
Semantic Retrieval
          │
          ▼
Relevant Context
          │
          ▼
Gemini 2.5 Flash
          │
          ▼
Generated Answer + Source Citations
```

---

# 🛠️ Tech Stack

| Category             | Technologies                       |
| -------------------- | ---------------------------------- |
| Programming Language | Python                             |
| LLM Framework        | LangChain                          |
| Embedding Model      | Hugging Face Sentence Transformers |
| Vector Database      | ChromaDB                           |
| Large Language Model | Google Gemini 2.5 Flash            |
| Frontend             | HTML, Tailwind CSS, JavaScript     |

---

# 📂 Project Structure

```text
RAG-Based-Document-Question-Answering-System/
│
└──static 
│    └──index.html
├── main.py
├──.gitignore
├── requirements.txt
└── README.md
```

---

# 🚀 How It Works

### Step 1 – Upload Documents

Upload one or more PDF or text documents through the web interface.

### Step 2 – Document Processing

The system extracts text and divides it into meaningful chunks.

### Step 3 – Embedding Generation

Each chunk is converted into a high-dimensional vector using Hugging Face embedding models.

### Step 4 – Vector Storage

Embeddings are stored in ChromaDB for efficient semantic retrieval.

### Step 5 – Ask Questions

Users ask questions in natural language.

### Step 6 – Context Retrieval

LangChain retrieves the most relevant document chunks using semantic similarity.

### Step 7 – Answer Generation

Gemini 2.5 Flash generates an accurate answer using only the retrieved context.

### Step 8 – Source References

The application displays the document chunks used to generate the response.

---

# 💡 Example

### User Question

> "What are the eligibility criteria mentioned in the document?"

### AI Response

* Minimum 2 years of experience
* Bachelor's degree in Computer Science
* Knowledge of Python and SQL
* Strong communication skills

**Referenced Sections**

* Page 2
* Page 5
