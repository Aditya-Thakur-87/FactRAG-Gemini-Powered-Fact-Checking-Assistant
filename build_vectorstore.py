# build_vectorstore.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from tqdm import tqdm
from config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, FAISS_STORE_DIR
import pandas as pd
from pathlib import Path

def build_rag_dataframe(rows):
    """
    rows: iterable of dicts containing at least 'id', 'claim', 'page_evidence'
    returns: pandas DataFrame
    """
    df = pd.DataFrame(rows)
    return df

def chunk_and_create_faiss(rag_df, output_dir: str = None):
    if output_dir is None:
        output_dir = FAISS_STORE_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                              chunk_overlap=CHUNK_OVERLAP,
                                              length_function=len)

    docs = []
    for _, row in tqdm(rag_df.iterrows(), total=len(rag_df), desc="Splitting text into chunks"):
        text = row.get("page_evidence", "")
        if not text:
            continue
        chunks = splitter.split_text(text)
        for chunk in chunks:
            docs.append({
                "id": row["id"],
                "claim": row.get("claim"),
                "chunk": chunk
            })

    lc_docs = [
        Document(page_content=d["chunk"],
                 metadata={"id": d["id"], "claim": d.get("claim")})
        for d in tqdm(docs, desc="Preparing LangChain Documents")
    ]

    embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(lc_docs, embedding_fn)
    vectorstore.save_local(str(output_dir))
    print(f"✅ FAISS vectorstore created and saved to {output_dir}")
    return vectorstore

def load_faiss(store_dir: str, embedding_fn=None):
    if embedding_fn is None:
        embedding_fn = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vs = FAISS.load_local(store_dir, embedding_fn, allow_dangerous_deserialization=True)
    print("Loaded FAISS store from", store_dir)
    return vs
