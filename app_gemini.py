"""
Streamlit Web App for RAG Agent using Google Gemini 2.5 Flash model
Allows users to upload documents and answer queries through a web interface
"""

import os
from pathlib import Path
from typing import List
import tempfile
import io

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Agent - Gemini",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Google Generative AI Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_ID = "gemini-2.5-flash"


def load_documents_from_files(uploaded_files) -> List[Document]:
    """Load documents from uploaded files"""
    documents = []
    
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.type == "application/pdf":
                # Handle PDF files
                pdf_bytes = uploaded_file.read()
                pdf_file = io.BytesIO(pdf_bytes)
                
                # Save to temporary file for PyPDFLoader
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_bytes)
                    tmp_path = tmp_file.name
                
                try:
                    pdf_loader = PyPDFLoader(tmp_path)
                    docs = pdf_loader.load()
                    
                    # Add source information
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name
                    
                    documents.extend(docs)
                finally:
                    os.remove(tmp_path)
            
            elif uploaded_file.type == "text/plain":
                # Handle text files
                text_content = uploaded_file.read().decode("utf-8")
                doc = Document(
                    page_content=text_content,
                    metadata={"source": uploaded_file.name}
                )
                documents.append(doc)
        
        except Exception as e:
            st.error(f"Error loading {uploaded_file.name}: {str(e)}")
    
    return documents


def create_vector_store(documents: List, embeddings) -> FAISS:
    """Create FAISS vector store from documents"""
    if not documents:
        raise ValueError("No documents found. Please add documents.")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create vector store
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore


def create_rag_chain(vectorstore, llm):
    """Create RAG chain with custom prompt"""
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Question: {question}

Answer:"""
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    qa_chain = (
        {"question": RunnablePassthrough()}
        | RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["question"])))
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain, retriever


@st.cache_resource
def initialize_models():
    """Initialize Google Generative AI models (cached)"""
    try:
        # Initialize embeddings using the latest model
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GOOGLE_API_KEY
        )
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model=MODEL_ID,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7,
            max_output_tokens=1000
        )
        
        return embeddings, llm
    except Exception as e:
        st.error(f"Failed to initialize Google Generative AI: {str(e)}")
        st.error("Please ensure GOOGLE_API_KEY is properly configured in .env file.")
        return None, None


@st.cache_resource
def load_and_process_documents():
    """Load documents and create vector store (cached)"""
    try:
        # Check if documents are in session state
        if "uploaded_files" not in st.session_state or not st.session_state.uploaded_files:
            return None, None, 0
        
        documents = load_documents_from_files(st.session_state.uploaded_files)
        
        if not documents:
            st.warning("No valid documents found in uploads.")
            return None, None, 0
        
        embeddings, llm = initialize_models()
        
        if embeddings is None or llm is None:
            return None, None, 0
        
        vectorstore = create_vector_store(documents, embeddings)
        qa_chain, retriever = create_rag_chain(vectorstore, llm)
        
        return qa_chain, retriever, len(documents)
    
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return None, None, 0


def main():
    """Main Streamlit app"""
    st.title("🚀 RAG Agent with Google Gemini")
    st.markdown("**Upload documents and ask questions about them**")
    
    # Check API key
    if not GOOGLE_API_KEY:
        st.error("❌ GOOGLE_API_KEY is not configured. Please add it to your .env file.")
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("📁 Document Upload")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload your documents (PDF or TXT)",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        # Store uploaded files in session state
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")
        
        if st.button("🔄 Reload Documents", key="reload_docs"):
            st.cache_resource.clear()
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        st.header("Settings")
        st.info(
            "This RAG Agent uses Google's Gemini 2.5 Flash model to answer questions "
            "based on documents you upload. Supports PDF and TXT files."
        )
    
    # Check if documents are uploaded
    if "uploaded_files" not in st.session_state or not st.session_state.uploaded_files:
        st.warning("📤 Please upload documents to get started. Use the file uploader in the sidebar.")
        return
    
    # Load models and documents
    qa_chain, retriever, doc_count = load_and_process_documents()
    
    if qa_chain is None or retriever is None:
        st.error("Failed to process documents. Please check the logs above.")
        return
    
    # Display status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents Loaded", doc_count)
    with col2:
        st.metric("Model", "Gemini 2.5 Flash")
    with col3:
        st.metric("Provider", "Google AI")
    
    st.divider()
    
    # Chat interface
    st.subheader("Ask Your Question")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            source_placeholder = st.empty()
            
            try:
                with st.spinner("Thinking..."):
                    # Get answer from RAG chain
                    #answer = qa_chain.invoke({"question": prompt})
                    answer = qa_chain.invoke(prompt)

                    
                    # Get source documents
                    sources = retriever.invoke(prompt)
                
                # Display answer
                message_placeholder.markdown(answer)
                
                # Display sources
                with source_placeholder.expander(f"📖 Sources ({len(sources)} documents)"):
                    for i, doc in enumerate(sources, 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                        if doc.metadata:
                            st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                
                # Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.error("Please check your Google API key configuration.")


if __name__ == "__main__":
    main()
