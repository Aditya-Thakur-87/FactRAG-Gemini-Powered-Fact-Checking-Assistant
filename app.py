import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
import google.generativeai as genai
from google.api_core.exceptions import PermissionDenied, InvalidArgument

# ----------------------------
# 🔧 Streamlit Page Setup
# ----------------------------
st.set_option('client.showErrorDetails', False)
st.set_page_config(page_title="FactRAG - Fact Checking App", page_icon="✅", layout="wide")
st.title("🧠 FactRAG — Gemini-Powered Fact-Checking Assistant")
st.caption("A Retrieval-Augmented Generation (RAG) app using LangChain + Gemini + FAISS + FEVER dataset.")

# ----------------------------
# 🔑 API Key Handling & Validation
# ----------------------------
if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state["GOOGLE_API_KEY"] = ""

api_key = st.text_input(
    "🔑 Enter your Google Gemini API key:",
    type="password",
    value=st.session_state.get("GOOGLE_API_KEY", "")
)

if not api_key.strip():
    st.warning("⚠️ Please enter your Google Gemini API key to continue.")
    st.stop()

# Validate key before proceeding
try:
    genai.configure(api_key=api_key)
    _ = genai.list_models()  # lightweight call
except (PermissionDenied, InvalidArgument, Exception):
    st.error("❌ Invalid Google API key. Please check your key and try again.")
    st.stop()

# Save valid key
st.session_state["GOOGLE_API_KEY"] = api_key

# ----------------------------
# ⚙️ Load Retriever (FAISS)
# ----------------------------
@st.cache_resource
def load_retriever():
    st.write("🔄 Loading retriever and embeddings...")
    embedding_fn = HuggingFaceEmbeddings(model_name="multi-qa-MiniLM-L6-cos-v1")
    vectorstore = FAISS.load_local("fever_faiss_store", embedding_fn, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    st.success("✅ Retriever loaded successfully.")
    return retriever

retriever = load_retriever()

# ----------------------------
# 🤖 Load Gemini LLM
# ----------------------------
@st.cache_resource
def load_llm(api_key):
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",  # use gemini-2.5-flash-lite if available
        temperature=0.2,
        max_output_tokens=512,
        google_api_key=api_key
    )

try:
    llm = load_llm(st.session_state["GOOGLE_API_KEY"])
except Exception:
    st.error("⚠️ Could not initialize Gemini model. Check your API key or internet connection.")
    st.stop()

# ----------------------------
# 🧠 Define Prompt Template
# ----------------------------
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

# ----------------------------
# 🔍 Context Retrieval
# ----------------------------
def retrieve_context(question: str) -> str:
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    return context

# ----------------------------
# ⚡ Build RAG Chain
# ----------------------------
parallel_chain = RunnableParallel({
    "question": RunnablePassthrough(),
    "context": RunnableLambda(retrieve_context)
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser

# ----------------------------
# 💬 Streamlit UI
# ----------------------------
query = st.text_area(
    "✍️ Enter a claim to fact-check:",
    placeholder="e.g., Nikolaj Coster-Waldau worked with Fox Broadcasting Company."
)

if st.button("🔍 Verify Claim") and query.strip():
    with st.spinner("Retrieving evidence and verifying claim..."):
        try:
            result = main_chain.invoke(query)
            docs = retriever.invoke(query)
        except Exception as e:
            st.error("⚠️ Something went wrong while verifying. Please try again.")
            st.stop()

    st.markdown("### 🧠 Model Verdict")
    if "SUPPORTS" in result.upper():
        st.success(result)
    elif "REFUTES" in result.upper():
        st.error(result)
    elif "NOT ENOUGH INFO" in result.upper():
        st.warning(result)
    else:
        st.info(result)

    st.markdown("### 📚 Top Retrieved Evidence")
    for i, doc in enumerate(docs, 1):
        with st.expander(f"Evidence {i}"):
            st.write(doc.page_content)
            if doc.metadata:
                st.json(doc.metadata)
else:
    st.info("💡 Enter a claim and click **Verify Claim** to begin.")
