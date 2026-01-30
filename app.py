"""
Streamlit Web App for RAG Agent using AWS Bedrock Qwen model
Reads documents from doc/ folder and answers user queries through a web interface
"""

import os
from pathlib import Path
from typing import List

import boto3
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws import BedrockEmbeddings, BedrockLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = "qwen.qwen3-32b-v1:0"

# Get API key from environment variable
BEDROCK_API_KEY = os.getenv("BEDROCK_API_KEY") or os.getenv("AWS_BEARER_TOKEN_BEDROCK")

# Set the API key as environment variable for boto3 to use automatically
if BEDROCK_API_KEY:
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = BEDROCK_API_KEY


def load_documents(doc_folder: str = "doc") -> List:
    """Load all documents from the doc folder"""
    doc_path = Path(doc_folder)
    
    if not doc_path.exists():
        doc_path.mkdir()
        return []
    
    documents = []
    
    # Load PDF files
    pdf_files = list(doc_path.glob("*.pdf"))
    if pdf_files:
        pdf_loader = DirectoryLoader(
            str(doc_path),
            glob="*.pdf",
            loader_cls=PyPDFLoader
        )
        documents.extend(pdf_loader.load())
    
    # Load text files
    txt_files = list(doc_path.glob("*.txt"))
    if txt_files:
        txt_loader = DirectoryLoader(
            str(doc_path),
            glob="*.txt",
            loader_cls=TextLoader
        )
        documents.extend(txt_loader.load())
    
    return documents


def create_vector_store(documents: List, embeddings) -> FAISS:
    """Create FAISS vector store from documents"""
    if not documents:
        raise ValueError("No documents found. Please add documents to doc/ folder.")
    
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
    """Initialize AWS Bedrock models (cached)"""
    try:
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=AWS_REGION
        )
        
        embeddings = BedrockEmbeddings(
            client=bedrock_runtime,
            model_id="amazon.titan-embed-text-v1"
        )
        
        llm = BedrockLLM(
            client=bedrock_runtime,
            model_id=MODEL_ID,
            model_kwargs={
                "temperature": 0.7,
                "max_tokens": 1000
            }
        )
        
        return embeddings, llm, bedrock_runtime
    except Exception as e:
        st.error(f"Failed to initialize AWS Bedrock: {str(e)}")
        st.error("Please ensure AWS credentials are properly configured.")
        return None, None, None


@st.cache_resource
def load_and_process_documents():
    """Load documents and create vector store (cached)"""
    try:
        documents = load_documents()
        
        if not documents:
            st.warning("No documents found in doc/ folder. Please add PDF or TXT files.")
            return None, None, 0
        
        embeddings, llm, _ = initialize_models()
        
        if embeddings is None:
            return None, None, 0
        
        vectorstore = create_vector_store(documents, embeddings)
        qa_chain, retriever = create_rag_chain(vectorstore, llm)
        
        return qa_chain, retriever, len(documents)
    
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return None, None, 0


def main():
    """Main Streamlit app"""
    st.title("📚 RAG Agent with AWS Bedrock")
    st.markdown("**Ask questions about your documents**")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        st.markdown("### Settings")
        
        if st.button("🔄 Reload Documents", key="reload_docs"):
            st.cache_resource.clear()
            st.rerun()
        
        doc_folder = st.text_input("Document Folder", value="doc", key="doc_folder")
        
        st.markdown("### About")
        st.info(
            "This RAG Agent uses AWS Bedrock's Qwen model to answer questions "
            "based on documents in your doc/ folder. Supports PDF and TXT files."
        )
    
    # Load models and documents
    qa_chain, retriever, doc_count = load_and_process_documents()
    
    if qa_chain is None or retriever is None:
        st.error("Failed to initialize the RAG agent. Please check the logs above.")
        return
    
    # Display status
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Documents Loaded", doc_count)
    with col2:
        st.metric("Model", "Qwen 3 32B")
    with col3:
        st.metric("Region", AWS_REGION)
    
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
                    answer = qa_chain.invoke({"question": prompt})
                    
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
                            st.caption(f"Metadata: {doc.metadata}")
                
                # Add assistant message to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.error("Please check your AWS credentials and configuration.")


if __name__ == "__main__":
    main()
