# rag_chat.py
from build_vectorstore import load_faiss
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from config import FAISS_STORE_DIR, GOOGLE_API_KEY
import os

def create_llm():
    if not GOOGLE_API_KEY:
        print("Warning: GOOGLE_API_KEY not set. LLM calls will fail unless set.")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.3,
        max_output_tokens=512
    )
    return llm

def create_retriever(store_dir: str = None):
    store_dir = store_dir or FAISS_STORE_DIR
    embedding_fn = HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-cos-V1")
    vs = load_faiss(store_dir, embedding_fn)
    retriever = vs.as_retriever(search_kwargs={"k": 6})
    return retriever

def build_chain(retriever, llm):
    parser = StrOutputParser()
    prompt = PromptTemplate(
        template="""
You are a fact-checking assistant.
Use the provided evidence to verify the claim and respond with one of:
- SUPPORTS
- REFUTES
- NOT ENOUGH INFO

Claim: {question}

Evidence:
{context}

Your answer and brief reasoning:
""",
        input_variables=["question", "context"],
    )

    def retrieve_context(question: str) -> str:
        docs = retriever.invoke(question)
        return "\n\n".join(d.page_content for d in docs)

    parallel_chain = RunnableParallel({
        "question": RunnablePassthrough(),
        "context": RunnableLambda(retrieve_context)
    })

    main_chain = parallel_chain | prompt | llm | parser
    return main_chain

def query_chain(main_chain, query: str):
    return main_chain.invoke(query)

if __name__ == "__main__":
    llm = create_llm()
    retriever = create_retriever()
    chain = build_chain(retriever, llm)
    query = "Source Code had Jake Gyllenhaal in it?"
    answer = query_chain(chain, query)
    print("Answer:\n", answer)
