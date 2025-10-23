# config.py
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# Project paths (change if needed)
ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
FEVER_DIR = DATA_DIR / "Fever"
WIKI_PAGES_DIR = FEVER_DIR / "wiki-pages"
FAISS_STORE_DIR = ROOT / "fever_faiss_store"

# Keys / tokens from environment     # HuggingFace token
GOOGLE_API_KEY = hf_your_google_apikey_here  # for Google Generative AI if used

# Embedding model — configure if you want another
EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-V1"

# FAISS parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
