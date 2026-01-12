# RAGLab 🧪  
### A Multi-Retriever RAG Experimentation & Evaluation Framework

RAGLab is a **Streamlit-based experimentation lab** for building, comparing, and evaluating **Retrieval-Augmented Generation (RAG)** pipelines on real documents.

It is designed to explore **how different retrieval strategies behave**, and to **measure their impact** using classical NLP metrics as well as LLM-based evaluation.

---

## 🚀 Key Features

### 📄 Document Ingestion
- Upload **PDF, TXT, Markdown, CSV, JSON**
- Adjustable **chunk size & overlap**
- Recursive text splitting for long documents

---

### 🔎 Retrieval Strategies
RAGLab supports multiple retrieval paradigms:

- **Vector Retrieval**
  - ChromaDB / FAISS
  - OpenAI or HuggingFace embeddings
- **BM25 (Keyword-based) Retrieval**
- **Ensemble Retrieval**
  - Weighted combination of vector + BM25
- **Contextual Compression Retrieval**
  - Redundancy filtering
  - Embedding-based clustering
- **HyDE (Hypothetical Document Embeddings)**
- **Agent-based Retrieval**
  - Tool usage + reasoning loop (ReAct-style)

Retrieval methods can be switched dynamically from the UI.

---

### 🧠 LLM Integration
- OpenAI chat models
- Query rewriting (history-aware)
- Context-grounded answering with source attribution
- Agent-driven document search

---

### 📊 Evaluation & Benchmarking
RAGLab includes a built-in evaluation suite:

- **BLEU**
- **ROUGE (1 / 2 / L)**
- **Semantic Similarity** (SentenceTransformers)
- **LLM-as-a-Judge Scoring** (0–10, normalized)

Evaluation results are:
- Stored in-session
- Visualized with charts
- Exportable as CSV

This allows **side-by-side comparison of retrieval strategies**.

---

### 🖥️ Interactive UI (Streamlit)
- Upload & process documents
- Configure embeddings and vector stores
- Select retrieval method
- Chat with documents
- Inspect retrieved chunks
- Analyze evaluation metrics visually

---

## 🧱 Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **ChromaDB / FAISS**
- **OpenAI API**
- **SentenceTransformers**
- **NLTK / ROUGE**
- **Pydantic**
- **Pandas / NumPy / Matplotlib**

---

## ▶️ How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

