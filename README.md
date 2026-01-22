
---

# 📚 Retrieval-Augmented Generation (RAG) from YouTube Transcripts

This project implements an **end-to-end Retrieval-Augmented Generation (RAG) pipeline** that extracts knowledge from a YouTube video transcript and answers user questions **strictly grounded in the video content** using vector search and a large language model. 

---

## 🚀 Project Overview

The system follows the standard **RAG architecture**:

1. **Document Ingestion** – Fetch YouTube video transcripts
2. **Chunking** – Split long transcripts into manageable text chunks
3. **Embedding & Indexing** – Convert chunks into vector embeddings and store them in FAISS
4. **Retrieval** – Retrieve the most relevant transcript chunks for a user query
5. **Augmentation** – Inject retrieved context into a controlled prompt
6. **Generation** – Generate answers using an LLM constrained to retrieved context

This approach **reduces hallucinations** and ensures responses are **context-faithful**.

---

## 🧠 Architecture Flow

```
YouTube Video
     ↓
Transcript Extraction
     ↓
Text Chunking
     ↓
Vector Embeddings (OpenAI)
     ↓
FAISS Vector Store
     ↓
Similarity Search (Top-K)
     ↓
Prompt Augmentation
     ↓
LLM Answer Generation
```

---

## 🧩 Key Components

### 1️⃣ Transcript Ingestion

* Uses `youtube-transcript-api` to fetch captions by **video ID**
* Supports English transcripts
* Gracefully handles videos with disabled captions 

---

### 2️⃣ Text Chunking

* Applies `RecursiveCharacterTextSplitter`
* Configuration:

  * `chunk_size = 1000`
  * `chunk_overlap = 200`
* Ensures semantic continuity across chunks

---

### 3️⃣ Embedding & Vector Storage

* Generates embeddings using **OpenAI embedding models**
* Stores embeddings in a **FAISS vector database**
* Enables fast semantic similarity search at scale 

---

### 4️⃣ Retrieval

* Converts FAISS index into a retriever
* Uses **cosine similarity**
* Retrieves top-K most relevant chunks (`k = 4`) per query

---

### 5️⃣ Prompt Augmentation

* Custom prompt template enforces **strict grounding**
* The model is instructed to:

  * Answer **only from retrieved transcript content**
  * Say *“I don’t know”* if information is missing

This ensures factual reliability and transparency.

---

### 6️⃣ Answer Generation

* Uses an OpenAI chat model with **low temperature**
* Produces concise, context-aware answers
* Prevents hallucination outside retrieved evidence

---

### 7️⃣ LangChain Pipeline (Composable Chains)

* Implements:

  * `RunnableParallel`
  * `RunnablePassthrough`
  * `RunnableLambda`
* Builds a reusable RAG chain supporting:

  * Direct Q&A
  * Video summarization
  * Named entity queries (e.g., “Who is Demis?”)

---

## 🛠️ Tech Stack

* **Python**
* **LangChain**
* **FAISS**
* **OpenAI Embeddings & Chat Models**
* **YouTube Transcript API**
* **tiktoken**
* **dotenv**

---

## ▶️ How to Run

```bash
pip install youtube-transcript-api langchain-community langchain-openai \
           faiss-cpu tiktoken python-dotenv
```

Set your API key:

```bash
export OPENAI_API_KEY="your_api_key"
```

Run the script:

```bash
python RAG.py
```

---


## 🔐 Design Strengths

* Retrieval-grounded answers
* Reduced hallucination risk
* Modular, extensible RAG pipeline
* Production-ready architecture pattern

---



