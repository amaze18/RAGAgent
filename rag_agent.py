"""
Simple RAG Agent using AWS Bedrock Qwen model
Reads documents from doc/ folder and answers user queries
"""

import os
from pathlib import Path
from typing import List

import boto3
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

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = "qwen.qwen3-32b-v1:0"

# Get API key from environment variable
BEDROCK_API_KEY = os.getenv("BEDROCK_API_KEY") or os.getenv("AWS_BEARER_TOKEN_BEDROCK")

# Set the API key as environment variable for boto3 to use automatically
# AWS Bedrock will automatically use AWS_BEARER_TOKEN_BEDROCK if set
if BEDROCK_API_KEY:
    os.environ["AWS_BEARER_TOKEN_BEDROCK"] = BEDROCK_API_KEY
    print("API key loaded and set as AWS_BEARER_TOKEN_BEDROCK")

# Initialize AWS Bedrock client
# boto3 will automatically use AWS_BEARER_TOKEN_BEDROCK if available
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name=AWS_REGION
)


def load_documents(doc_folder: str = "doc") -> List:
    """Load all documents from the doc folder"""
    doc_path = Path(doc_folder)
    
    if not doc_path.exists():
        print(f"Creating {doc_folder} folder...")
        doc_path.mkdir()
        print(f"Please add documents to {doc_folder} folder and run again.")
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
    
    print(f"Created {len(splits)} document chunks")
    
    # Create vector store
    vectorstore = FAISS.from_documents(splits, embeddings)
    print("Vector store created successfully")
    
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
    
    def format_prompt(question):
        docs = retriever.invoke(question)
        context = format_docs(docs)
        return prompt.format(context=context, question=question)
    
    qa_chain = (
        {"question": RunnablePassthrough()}
        | RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["question"])))
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return qa_chain, retriever


def main():
    """Main function to run the RAG agent"""
    print("=" * 60)
    print("RAG Agent with AWS Bedrock Qwen Model")
    print("=" * 60)
    
    # Initialize embeddings and LLM
    print("\nInitializing AWS Bedrock models...")
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
    
    # Load documents
    print("\nLoading documents from doc/ folder...")
    documents = load_documents()
    
    if not documents:
        print("No documents found. Exiting.")
        return
    
    print(f"Loaded {len(documents)} documents")
    
    # Create vector store
    print("\nCreating vector store...")
    vectorstore = create_vector_store(documents, embeddings)
    
    # Create RAG chain
    print("\nCreating RAG chain...")
    qa_chain, retriever = create_rag_chain(vectorstore, llm)
    
    # Interactive Q&A loop
    print("\n" + "=" * 60)
    print("RAG Agent is ready! Ask your questions.")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 60 + "\n")
    
    while True:
        query = input("Your question: ").strip()
        
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        try:
            print("\nThinking...")
            answer = qa_chain.invoke({"question": query})
            sources = retriever.invoke(query)
            
            print(f"\nAnswer: {answer}")
            print(f"\nSources: {len(sources)} documents used")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Please check your AWS credentials and region configuration.")
        
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()

