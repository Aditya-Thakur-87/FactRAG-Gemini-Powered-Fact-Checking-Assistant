# 🧠 FactRAG: Gemini-Powered Fact-Checking Assistant

A **Retrieval-Augmented Generation (RAG)** chatbot designed to fact-check claims using **Wikipedia evidence** and **Google Gemini (Generative AI)**.  
This project uses **LangChain**, **FAISS**, and **Hugging Face embeddings** to retrieve and verify factual information, inspired by the **FEVER dataset** structure.

---

## 🚀 Features

- 🧩 **RAG-based pipeline**: Combines retrieval (FAISS + HuggingFaceEmbeddings) with generation (Gemini).
- 🔍 **Wikipedia-based evidence search** via FEVER-style `wiki-pages` data.
- 💬 **Fact verification chatbot** using Gemini 2.5 Flash Lite model.
- 🧠 **Custom vector store builder** with chunking for efficient retrieval.
- ⚡ Modular code: each function separated into a clear, reusable module.
- 🔒 Secrets handled safely using `.env` file (no hardcoded tokens).

---

## 📁 Project Structure

```
Fact_check_RAG_chatbot/
│
├── app.py                        # Optional Streamlit app (for UI)
├── build_vectorstore.py           # Builds FAISS store from Wikipedia text chunks
├── config.py                      # Global configs, paths, model names, API keys
├── data_conversion.py             # Converts & normalizes FEVER-like datasets
├── evidence.py                    # Evaluates & extracts evidence sentences/pages
├── fever_wiki.py                  # Loads & cleans FEVER Wikipedia data
├── hf_utils.py                    # Hugging Face utilities (whoami, etc.)
├── rag_chat.py                    # Core Gemini-powered RAG chatbot logic
├── utils.py                       # Helper text cleaning functions
├── requirements.txt               # Python dependencies
├── .env.example                   # Example environment file (fill your tokens here)
└── README.md                      # Project documentation
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Aditya-Thakur-87/FactRAG-Gemini-Powered-Fact-Checking-Assistant.git
cd FactRAG-Gemini-Powered-Fact-Checking-Assistant
```

### 2️⃣ Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate      # (Linux/Mac)
venv\Scripts\activate       # (Windows)
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Create `.env` file from example
```bash
cp .env.example .env
```

Then open `.env` and set your keys:
```
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

---

## 🧠 Workflow Overview

### 🔹 1. Data Loading
Load the FEVER dataset and Wikipedia dumps using `data_conversion.py` and `fever_wiki.py`.

### 🔹 2. Data Cleaning & Evidence Extraction
Clean wiki pages and extract claim-evidence pairs via `evidence.py`.

### 🔹 3. Build FAISS Vectorstore
Split Wikipedia text into overlapping chunks and embed them using Hugging Face sentence models.
```bash
python build_vectorstore.py
```

### 🔹 4. Run Fact-Check Chatbot
Use Gemini + FAISS retriever pipeline to verify claims.
```bash
python rag_chat.py
```

Example Query:
```
Claim: "Source Code had Jake Gyllenhaal in it?"
→ Output: SUPPORTS (with brief reasoning)
```

---

## 🧩 Key Components

### `build_vectorstore.py`
- Uses `RecursiveCharacterTextSplitter` for chunking Wikipedia text.  
- Embeds chunks using `HuggingFaceEmbeddings`.  
- Saves FAISS index locally for efficient retrieval.

### `rag_chat.py`
- Loads FAISS retriever and Google Gemini LLM.  
- Uses a **LangChain pipeline** (`PromptTemplate` → `Retriever` → `Gemini LLM`).  
- Classifies each claim as `SUPPORTS`, `REFUTES`, or `NOT ENOUGH INFO`.

### `fever_wiki.py`
- Loads all FEVER Wikipedia JSONL files.  
- Cleans noise tokens like `-LRB-`, `-RRB-`, etc.  
- Returns structured sentence dictionaries for fast lookup.

---

## 💡 Example Flow

```
1️⃣ Input Claim: "The Great Wall of China is visible from space."
2️⃣ Retriever: Finds Wikipedia sentences about the Great Wall.
3️⃣ LLM: Gemini verifies evidence → returns "REFUTES"
```

---

## 📦 Example `.env`
```
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXX
```

Keep this file private — **never commit it** to GitHub!

---

## 🧰 Requirements

- Python 3.9+  
- Dependencies listed in `requirements.txt`:
  ```
  datasets
  pandas
  tqdm
  langchain_huggingface
  langchain_text_splitters
  langchain_community
  langchain_core
  huggingface_hub
  python-dotenv
  faiss-cpu
  langchain_google_genai
  ```

(Use `faiss-gpu` if you have CUDA support.)

---

## ⚙️ Advanced Usage

You can integrate this project with **Streamlit** for a web UI:

```bash
streamlit run app.py
```

Or adapt it for an **API backend** with FastAPI / Flask.

---

## 🧩 Future Improvements

- 🔍 Add NLI (Natural Language Inference) layer for contradiction detection  
- ⚡ Improve retrieval ranking with hybrid search (BM25 + embeddings)  
- 🧠 Fine-tune on FEVER dataset for better factual reasoning  
- 🪶 Add response confidence scores and source citations  

---

