import streamlit as st
from pypdf import PdfReader
import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import google.generativeai as genai


# -------------------- CONFIG --------------------

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

st.set_page_config(
    page_title="PDF AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.header("🤖 PDF AI Assistant")
st.caption("Upload PDFs and chat with them intelligently.")


# -------------------- SESSION MEMORY --------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False


# -------------------- PDF TEXT EXTRACTION --------------------

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# -------------------- TEXT CHUNKING --------------------

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


# -------------------- VECTOR STORE --------------------

def create_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")
    st.session_state.vector_ready = True


# -------------------- RETRIEVAL CHAIN --------------------

def get_retrieval_chain():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful AI assistant.

    Answer the question based only on the provided context.
    If the answer is not available in the context, say:
    "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}
    """)

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# -------------------- SIDEBAR --------------------

with st.sidebar:
    st.title("📂 PDF Manager")

    pdf_docs = st.file_uploader(
        "Upload PDF Files",
        accept_multiple_files=True
    )

    if st.button("🔄 Process PDFs"):
        if not pdf_docs:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                create_vector_store(text_chunks)
                st.success("✅ PDFs processed successfully!")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# -------------------- CHAT UI --------------------

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
if prompt := st.chat_input("Ask something about your PDFs..."):

    if not st.session_state.vector_ready:
        st.warning("⚠️ Please upload and process PDFs first.")
    else:
        # Show user message
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        chain = get_retrieval_chain()
        response = chain.invoke(prompt)

        # Show assistant response
        with st.chat_message("assistant"):
            st.markdown(response)

        # Save assistant response
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
