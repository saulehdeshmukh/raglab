# Advanced RAG System with Multiple Retrieval Methods and Evaluations
# app.py

import os
import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import json
from typing import List, Dict, Any, Tuple
import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import logging
from pathlib import Path

# OpenAI imports
import openai
from openai import OpenAI

# Vector DB and embedding imports
import chromadb
from chromadb.utils import embedding_functions
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader, 
    UnstructuredMarkdownLoader,
    CSVLoader,
    JSONLoader
)

# Retrieval methods
from langchain.retrievers import (
    BM25Retriever,
    EnsembleRetriever,
    ContextualCompressionRetriever,
    ParentDocumentRetriever
)
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_community.document_transformers import (
    EmbeddingsRedundantFilter,
    EmbeddingsClusteringFilter
)

# For agents
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser

# For chains and query augmentation
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

# HyDE (Hypothetical Document Embeddings)
from langchain_community.retrievers import HypotheticalDocumentEmbedder


# For query expansion and rewriting
from langchain.chains import LLMChain, StuffDocumentsChain

# For memory management
from langchain.memory import ConversationBufferMemory

# Initialize NLTK downloads
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
TEMP_DIR = Path(tempfile.gettempdir()) / "advanced_rag_demo"
TEMP_DIR.mkdir(exist_ok=True)
CHROMA_DB_PATH = TEMP_DIR / "chroma_db"
FAISS_DB_PATH = TEMP_DIR / "faiss_index"
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context. 
If you don't know the answer or can't find it in the provided context, say so instead of making up information.
Always cite your sources by referencing the specific part of the context you used."""
DEFAULT_QUERY_REWRITE_PROMPT = """Given the conversation history and a question, rewrite the question to be a standalone question 
that captures all relevant context from the conversation history.
Original question: {question}
Conversation history: {chat_history}
Rewritten question:"""

# Streamlit setup
st.set_page_config(
    page_title="Advanced RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Helper functions
def format_docs(docs):
    """Format docs for prompt template."""
    return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

def get_token_count(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens for the given text and model."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def calculate_semantic_similarity(response: str, ground_truth: str, model="all-MiniLM-L6-v2") -> float:
    """Calculate semantic similarity between response and ground truth."""
    embedding_model = SentenceTransformer(model)
    resp_embedding = embedding_model.encode([response])
    truth_embedding = embedding_model.encode([ground_truth])
    return cosine_similarity(resp_embedding, truth_embedding)[0][0]

def calculate_rouge_scores(response: str, ground_truth: str) -> Dict[str, float]:
    """Calculate ROUGE scores between response and ground truth."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, response)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

def calculate_bleu_score(response: str, ground_truth: str) -> float:
    """Calculate BLEU score between response and ground truth."""
    smoothie = SmoothingFunction().method1
    response_tokens = nltk.word_tokenize(response.lower())
    ground_truth_tokens = [nltk.word_tokenize(ground_truth.lower())]
    return sentence_bleu(ground_truth_tokens, response_tokens, smoothing_function=smoothie)

def get_llm_eval_score(response: str, ground_truth: str, query: str, client) -> float:
    """Use LLM to evaluate relevance and correctness."""
    eval_prompt = f"""
    You are evaluating the quality of an AI assistant's response to a user query.
    
    User Query: {query}
    
    Ground Truth Answer: {ground_truth}
    
    AI Response: {response}
    
    Rate the response on a scale of 0 to 10, where:
    - 0-2: Incorrect or misleading
    - 3-5: Partially correct with significant omissions or inaccuracies
    - 6-8: Mostly correct with minor issues
    - 9-10: Fully correct and comprehensive
    
    Consider:
    1. Factual accuracy compared to the ground truth
    2. Completeness of the answer
    3. Relevance to the original query
    4. Clarity and coherence
    
    Provide your rating as a single number.
    """
    
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0
    )
    
    score_text = completion.choices[0].message.content.strip()
    # Extract just the numeric score with regex
    score_match = re.search(r'(\d+(\.\d+)?)', score_text)
    if score_match:
        return float(score_match.group(1))
    return 0.0  # Default if parsing fails

def evaluate_response(response: str, ground_truth: str, query: str, client) -> Dict[str, float]:
    """Evaluate response using multiple metrics."""
    return {
        'bleu': calculate_bleu_score(response, ground_truth),
        'rouge': calculate_rouge_scores(response, ground_truth),
        'semantic_similarity': calculate_semantic_similarity(response, ground_truth),
        'llm_score': get_llm_eval_score(response, ground_truth, query, client) / 10.0  # Normalize to 0-1
    }

def load_document(file, file_type=None):
    """Load document from file."""
    if file is None:
        return None
    
    # Save the uploaded file to a temporary file
    temp_file_path = os.path.join(TEMP_DIR, file.name)
    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())
    
    # Determine file type if not specified
    if file_type is None:
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension in ['.pdf']:
            file_type = 'pdf'
        elif file_extension in ['.txt']:
            file_type = 'text'
        elif file_extension in ['.md']:
            file_type = 'markdown'
        elif file_extension in ['.csv']:
            file_type = 'csv'
        elif file_extension in ['.json']:
            file_type = 'json'
        else:
            file_type = 'text'  # Default
    
    # Load document based on file type
    try:
        if file_type == 'pdf':
            loader = PyPDFLoader(temp_file_path)
        elif file_type == 'text':
            loader = TextLoader(temp_file_path)
        elif file_type == 'markdown':
            loader = UnstructuredMarkdownLoader(temp_file_path)
        elif file_type == 'csv':
            loader = CSVLoader(temp_file_path)
        elif file_type == 'json':
            def json_extract(record: dict, metadata: dict) -> str:
                text = record.get("text", "")
                return text
            loader = JSONLoader(
                file_path=temp_file_path,
                jq_schema='.[]',
                content_key="text"
            )
        
        documents = loader.load()
        return documents
    except Exception as e:
        logger.error(f"Error loading document: {e}")
        st.error(f"Error loading document: {e}")
        return None

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks."""
    if not documents:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    return text_splitter.split_documents(documents)

def create_embeddings_model(api_key=None, embed_model="text-embedding-3-small", use_openai=True):
    """Create embeddings model."""
    if use_openai and api_key:
        return OpenAIEmbeddings(
            model=embed_model,
            openai_api_key=api_key,
            show_progress_bar=True
        )
    else:
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_db(documents, embedding_model, db_type="chroma", collection_name="advanced_rag_demo"):
    """Create vector database."""
    if not documents:
        return None
    
    if db_type == "chroma":
        # Create a new Chroma collection
        db = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=str(CHROMA_DB_PATH)
        )
        db.persist()
        return db
    elif db_type == "faiss":
        db = FAISS.from_documents(
            documents=documents,
            embedding=embedding_model
        )
        # Save FAISS index
        db.save_local(str(FAISS_DB_PATH))
        return db
    else:
        raise ValueError(f"Unsupported vector DB type: {db_type}")

def create_bm25_retriever(documents, k=4):
    """Create BM25 retriever."""
    if not documents:
        return None
    
    # Extract texts from documents
    texts = [doc.page_content for doc in documents]
    
    # Create BM25 retriever
    return BM25Retriever.from_texts(texts, metadatas=[doc.metadata for doc in documents], k=k)

def create_ensemble_retriever(vector_retriever, bm25_retriever, weights=None):
    """Create ensemble retriever."""
    if not vector_retriever or not bm25_retriever:
        return None
    
    if weights is None:
        weights = [0.5, 0.5]  # Equal weights by default
    
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=weights
    )

def create_compression_retriever(base_retriever, embedding_model):
    """Create compression retriever that filters redundant results."""
    if not base_retriever:
        return None
    
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding_model)
    cluster_filter = EmbeddingsClusteringFilter(embeddings=embedding_model)
    
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[redundant_filter, cluster_filter]
    )
    
    return ContextualCompressionRetriever(
        base_compressor=pipeline_compressor,
        base_retriever=base_retriever
    )

def create_hyde_retriever(llm, embedding_model, vector_db, prompt_template=None):
    """Create HyDE (Hypothetical Document Embeddings) retriever."""
    if not embedding_model or not vector_db:
        return None
    
    if prompt_template is None:
        prompt_template = """
        Please write a passage that answers the question: {question}
        
        Passage:
        """
    
    return HypotheticalDocumentEmbedder.from_llm(
        llm=llm,
        base_embeddings=embedding_model,
        prompt_template=prompt_template
    ).embed_query

def create_parent_document_retriever(
    vector_db, 
    embedding_model, 
    parent_documents, 
    child_documents
):
    """Create parent document retriever that returns full parent documents."""
    if not vector_db or not embedding_model:
        return None
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    return ParentDocumentRetriever(
        vectorstore=vector_db,
        parent_documents=parent_documents,
        child_splitter=text_splitter,
        embedding=embedding_model
    )

def rewrite_query(query, chat_history, llm):
    """Rewrite query based on chat history."""
    prompt = ChatPromptTemplate.from_template(DEFAULT_QUERY_REWRITE_PROMPT)
    
    formatted_chat_history = "\n".join([
        f"Human: {q}\nAI: {a}" for q, a in chat_history
    ])
    
    chain = prompt | llm
    
    rewritten_query = chain.invoke({
        "question": query,
        "chat_history": formatted_chat_history
    })
    
    return rewritten_query.content

class SearchResult(BaseModel):
    """Search result schema for agents."""
    answer: str = Field(description="The answer to the query")
    source_documents: List[str] = Field(description="The source documents used to answer the query")

def create_search_agent(retriever, llm):
    """Create a search agent."""
    if not retriever or not llm:
        return None
    
    # Define search tool
    def search_documents(query: str) -> Dict[str, Any]:
        """Search documents using retriever."""
        docs = retriever.get_relevant_documents(query)
        answer_prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:
            {context}
            
            Question: {question}
            Answer:"""
        )
        
        chain = answer_prompt | llm
        
        answer = chain.invoke({
            "context": format_docs(docs),
            "question": query
        })
        
        return {
            "answer": answer.content,
            "source_documents": [doc.page_content for doc in docs]
        }
    
    # Create search tool
    search_tool = Tool(
        name="search_documents",
        description="Search documents for relevant information",
        func=search_documents
    )
    
    # Create output parser
    output_parser = PydanticOutputParser(pydantic_object=SearchResult)
    
    # Create agent prompt
    agent_prompt = PromptTemplate(
        template="""You are a research assistant. Use the search_documents tool to find information and answer the question.

Question: {query}

{agent_scratchpad}""",
        input_variables=["query", "agent_scratchpad"]
    )
    
    # Create agent
    agent = create_react_agent(
        llm=llm,
        tools=[search_tool],
        prompt=agent_prompt
    )
    
    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=[search_tool],
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

def setup_retrieval_chain(retriever, llm, system_prompt=None):
    """Setup retrieval chain with context."""
    if not retriever or not llm:
        return None
    
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])
    
    # Create chain
    return {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    } | prompt | llm

def setup_history_aware_retrieval(llm, retriever, memory=None):
    """Setup history-aware retrieval."""
    if not llm or not retriever:
        return None
    
    if memory is None:
        memory = ConversationBufferMemory(
            return_messages=True,
            output_key="answer",
            input_key="question"
        )
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    
    # Create prompt for the final LLM chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", DEFAULT_SYSTEM_PROMPT),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])
    
    # Chain to retrieve documents
    retrieval_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=StuffDocumentsChain(
            llm_chain=LLMChain(
                llm=llm,
                prompt=prompt
            ),
            document_variable_name="context"
        )
    )
    
    return retrieval_chain

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "documents" not in st.session_state:
        st.session_state.documents = None
    
    if "chunks" not in st.session_state:
        st.session_state.chunks = None
    
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    if "retriever_type" not in st.session_state:
        st.session_state.retriever_type = "vector"
    
    if "llm" not in st.session_state:
        st.session_state.llm = None
    
    if "chain" not in st.session_state:
        st.session_state.chain = None
    
    if "eval_results" not in st.session_state:
        st.session_state.eval_results = []
    
    if "openai_client" not in st.session_state:
        st.session_state.openai_client = None

# Main App UI
def main():
    init_session_state()
    
    st.title("🔍 Advanced RAG System with Multiple Retrieval Methods")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # API Key input
        api_key = st.text_input("OpenAI API Key", type="password", key="openai_api_key")
        
        if api_key:
            st.session_state.api_key = api_key
            st.session_state.openai_client = OpenAI(api_key=api_key)
            
            # Initialize LLM if API key is provided
            st.session_state.llm = ChatOpenAI(
                model_name="gpt-4o",
                openai_api_key=api_key,
                temperature=0.2
            )
        
        st.divider()
        
        # Document upload
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=["pdf", "txt", "md", "csv", "json"],
            accept_multiple_files=True
        )
        
        chunk_size = st.slider("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
        
        embedding_model = st.selectbox(
            "Embedding Model",
            options=["text-embedding-3-small", "text-embedding-3-large", "all-MiniLM-L6-v2"],
            index=0
        )
        
        use_openai = st.checkbox("Use OpenAI Embeddings", value=True)
        
        vector_db_type = st.selectbox(
            "Vector Database",
            options=["chroma", "faiss"],
            index=0
        )
        
        process_btn = st.button("Process Documents")
        
        if process_btn and uploaded_files and st.session_state.api_key:
            with st.spinner("Processing documents..."):
                # Load documents
                all_documents = []
                for file in uploaded_files:
                    docs = load_document(file)
                    if docs:
                        all_documents.extend(docs)
                
                if all_documents:
                    st.session_state.documents = all_documents
                    
                    # Split documents
                    st.session_state.chunks = split_documents(
                        all_documents,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    # Create embeddings model
                    emb_model = create_embeddings_model(
                        api_key=st.session_state.api_key,
                        embed_model=embedding_model,
                        use_openai=use_openai
                    )
                    
                    # Create vector database
                    st.session_state.vector_db = create_vector_db(
                        st.session_state.chunks,
                        emb_model,
                        db_type=vector_db_type
                    )
                    
                    # Create retrievers
                    vector_retriever = st.session_state.vector_db.as_retriever(
                        search_kwargs={"k": 4}
                    )
                    
                    bm25_retriever = create_bm25_retriever(st.session_state.chunks)
                    
                    # Store retrievers
                    st.session_state.vector_retriever = vector_retriever
                    st.session_state.bm25_retriever = bm25_retriever
                    st.session_state.ensemble_retriever = create_ensemble_retriever(
                        vector_retriever, bm25_retriever
                    )
                    st.session_state.compression_retriever = create_compression_retriever(
                        vector_retriever, emb_model
                    )
                    
                    if st.session_state.llm:
                        st.session_state.hyde_retriever_func = create_hyde_retriever(
                            st.session_state.llm, emb_model, st.session_state.vector_db
                        )
                    
                    # Default retriever
                    st.session_state.retriever = vector_retriever
                    st.session_state.retriever_type = "vector"
                    
                    # Create chain
                    st.session_state.chain = setup_retrieval_chain(
                        st.session_state.retriever,
                        st.session_state.llm
                    )
                    
                    # Create agent
                    st.session_state.agent = create_search_agent(
                        st.session_state.retriever,
                        st.session_state.llm
                    )
                    
                    st.success(f"Processed {len(all_documents)} documents into {len(st.session_state.chunks)} chunks!")
                    
                    # Show document statistics
                    st.text(f"Total documents: {len(all_documents)}")
                    st.text(f"Total chunks: {len(st.session_state.chunks)}")
                    st.text(f"Average chunk size: {sum(len(chunk.page_content) for chunk in st.session_state.chunks) // len(st.session_state.chunks)} chars")
        
        st.divider()
        
        # Retrieval method selection
        st.subheader("Retrieval Method")
        retriever_type = st.selectbox(
            "Select Retrieval Method",
            options=[
                "vector", 
                "bm25", 
                "ensemble", 
                "compression", 
                "hyde",
                "agent"
            ],
            index=0
        )
        
        # Apply button for retrieval method
        apply_retriever = st.button("Apply Retrieval Method")
        
        if apply_retriever:
            st.session_state.retriever_type = retriever_type
            
            if retriever_type == "vector" and hasattr(st.session_state, "vector_retriever"):
                st.session_state.retriever = st.session_state.vector_retriever
            elif retriever_type == "bm25" and hasattr(st.session_state, "bm25_retriever"):
                st.session_state.retriever = st.session_state.bm25_retriever
            elif retriever_type == "ensemble" and hasattr(st.session_state, "ensemble_retriever"):
                st.session_state.retriever = st.session_state.ensemble_retriever
            elif retriever_type == "compression" and hasattr(st.session_state, "compression_retriever"):
                st.session_state.retriever = st.session_state.compression_retriever
            elif retriever_type == "hyde" and hasattr(st.session_state, "hyde_retriever_func"):
                # For HyDE, we directly use the query transformation function
                pass  # We'll handle HyDE specially in the query processing
            elif retriever_type == "agent" and hasattr(st.session_state, "agent"):
                # For agent, we use the agent executor
                pass  # We'll handle agent specially in the query processing
            
            # If we have a valid retriever and LLM, create/update chain
            if st.session_state.retriever and st.session_state.llm and retriever_type != "agent" and retriever_type != "hyde":
                st.session_state.chain = setup_retrieval_chain(
                    st.session_state.retriever,
                    st.session_state.llm
                )
                st.success(f"Applied {retriever_type} retrieval method!")
            elif retriever_type == "agent" and hasattr(st.session_state, "agent"):
                st.success(f"Applied agent-based retrieval!")
            elif retriever_type == "hyde" and hasattr(st.session_state, "hyde_retriever_func"):
                st.success(f"Applied HyDE retrieval method!")
        
        st.divider()
        
        # Evaluation section
        st.subheader("Evaluation")
        
        ground_truth = st.text_area("Ground Truth (for evaluation)", height=100)
        eval_query = st.text_input("Evaluation Query")
        
        run_eval = st.button("Run Evaluation")
        
        if run_eval and ground_truth and eval_query and st.session_state.retriever and st.session_state.llm:
            with st.spinner("Running evaluation..."):
                response = ""
                
                if st.session_state.retriever_type == "agent":
                    agent_result = st.session_state.agent.invoke({"query": eval_query})
                    response = agent_result.get("output", "")
                elif st.session_state.retriever_type == "hyde" and st.session_state.hyde_retriever_func:
                    # For HyDE, we create a one-time query
                    hyde_embedding = st.session_state.hyde_retriever_func(eval_query)
                    docs = st.session_state.vector_db.similarity_search_by_vector(hyde_embedding, k=4)
                    
                    context = format_docs(docs)
                    
                    # Create a temporary chain for this query
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", DEFAULT_SYSTEM_PROMPT),
                        ("human", f"Context:\n{context}\n\nQuestion: {eval_query}")
                    ])
                    
                    chain = prompt | st.session_state.llm
                    
                    response_message = chain.invoke({})
                    response = response_message.content
                else:
                    # For regular retrievers
                    chain_result = st.session_state.chain.invoke(eval_query)
                    response = chain_result.content
                
                # Evaluate response
                eval_results = evaluate_response(
                    response,
                    ground_truth,
                    eval_query,
                    st.session_state.openai_client
                )
                
                # Store evaluation results
                new_eval = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "retriever": st.session_state.retriever_type,
                    "query": eval_query,
                    "response": response,
                    "ground_truth": ground_truth,
                    "bleu": eval_results["bleu"],
                    "rouge1": eval_results["rouge"]["rouge1"],
                    "rouge2": eval_results["rouge"]["rouge2"],
                    "rougeL": eval_results["rouge"]["rougeL"],
                    "semantic_similarity": eval_results["semantic_similarity"],
                    "llm_score": eval_results["llm_score"]
                }
                
                st.session_state.eval_results.append(new_eval)
                st.success("Evaluation complete!")
    
    # Main content area - Chat interface
    tabs = st.tabs(["Chat", "Document Explorer", "Evaluation Results"])
    
    # Chat tab
    with tabs[0]:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if query := st.chat_input("Ask a question about your documents"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": query})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(query)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    if not st.session_state.retriever and st.session_state.retriever_type != "agent" and st.session_state.retriever_type != "hyde":
                        response = "Please upload and process documents first."
                    elif not st.session_state.llm:
                        response = "Please provide an OpenAI API key first."
                    else:
                        try:
                            if st.session_state.retriever_type == "agent":
                                # Use agent for retrieval and response
                                result = st.session_state.agent.invoke({"query": query})
                                response = result.get("output", "I couldn't find a relevant answer in the documents.")
                            elif st.session_state.retriever_type == "hyde" and hasattr(st.session_state, "hyde_retriever_func"):
                                # For HyDE, generate a hypothetical document embedding
                                hyde_embedding = st.session_state.hyde_retriever_func(query)
                                docs = st.session_state.vector_db.similarity_search_by_vector(hyde_embedding, k=4)
                                
                                # Create a temporary chain
                                context = format_docs(docs)
                                
                                prompt = ChatPromptTemplate.from_messages([
                                    ("system", DEFAULT_SYSTEM_PROMPT),
                                    ("human", f"Context:\n{context}\n\nQuestion: {query}")
                                ])
                                
                                chain = prompt | st.session_state.llm
                                
                                response_message = chain.invoke({})
                                response = response_message.content
                                
                                # Show retrieved docs
                                with st.expander("Retrieved Documents"):
                                    for i, doc in enumerate(docs):
                                        st.markdown(f"**Document {i+1}**")
                                        st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                        st.divider()
                            else:
                                # Use regular chain for response
                                result = st.session_state.chain.invoke(query)
                                response = result.content
                                
                                # Show retrieved docs
                                if hasattr(st.session_state.retriever, "get_relevant_documents"):
                                    docs = st.session_state.retriever.get_relevant_documents(query)
                                    with st.expander("Retrieved Documents"):
                                        for i, doc in enumerate(docs):
                                            st.markdown(f"**Document {i+1}**")
                                            st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                                            st.divider()
                            
                            # Update chat history for context-aware retrieval
                            st.session_state.chat_history.append((query, response))
                        except Exception as e:
                            response = f"Error: {str(e)}"
                            st.error(f"Error generating response: {str(e)}")
                
                st.markdown(response)
            
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Document Explorer tab
    with tabs[1]:
        st.header("Document Explorer")
        
        if st.session_state.documents:
            st.subheader("Original Documents")
            for i, doc in enumerate(st.session_state.documents):
                with st.expander(f"Document {i+1} - {doc.metadata.get('source', 'Unknown source')}"):
                    st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
            
            st.subheader("Document Chunks")
            for i, chunk in enumerate(st.session_state.chunks[:10]):  # Limit to first 10 chunks
                with st.expander(f"Chunk {i+1} - {chunk.metadata.get('source', 'Unknown source')}"):
                    st.text(chunk.page_content)
            
            if len(st.session_state.chunks) > 10:
                st.info(f"Showing 10 out of {len(st.session_state.chunks)} chunks.")
        else:
            st.info("No documents uploaded. Please upload documents in the sidebar.")
    
    # Evaluation Results tab
    with tabs[2]:
        st.header("Evaluation Results")
        
        if st.session_state.eval_results:
            # Create DataFrame from evaluation results
            df = pd.DataFrame(st.session_state.eval_results)
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg BLEU Score", f"{df['bleu'].mean():.3f}")
            
            with col2:
                st.metric("Avg ROUGE-L", f"{df['rougeL'].mean():.3f}")
            
            with col3:
                st.metric("Avg Semantic Similarity", f"{df['semantic_similarity'].mean():.3f}")
            
            with col4:
                st.metric("Avg LLM Score", f"{df['llm_score'].mean():.3f}")
            
            # Plot evaluation results
            st.subheader("Evaluation Metrics by Retriever Type")
            
            # Group by retriever type
            grouped = df.groupby("retriever").mean(numeric_only=True)
            
            # Plot metrics
            fig, ax = plt.subplots(figsize=(10, 6))
            
            metrics = ["bleu", "rougeL", "semantic_similarity", "llm_score"]
            labels = ["BLEU", "ROUGE-L", "Semantic Similarity", "LLM Score"]
            
            x = np.arange(len(grouped.index))
            width = 0.2
            
            for i, metric in enumerate(metrics):
                ax.bar(x + i*width - 0.3, grouped[metric], width, label=labels[i])
            
            ax.set_xlabel("Retriever Type")
            ax.set_ylabel("Score")
            ax.set_title("Evaluation Metrics by Retriever Type")
            ax.set_xticks(x)
            ax.set_xticklabels(grouped.index)
            ax.legend()
            
            st.pyplot(fig)
            
            # Display detailed results in table
            st.subheader("Detailed Evaluation Results")
            st.dataframe(
                df[["timestamp", "retriever", "query", "bleu", "rouge1", "rouge2", "rougeL", "semantic_similarity", "llm_score"]],
                use_container_width=True
            )
            
            # Allow downloading of results
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Evaluation Results",
                data=csv,
                file_name="rag_evaluation_results.csv",
                mime="text/csv"
            )
        else:
            st.info("No evaluation results yet. Run evaluations in the sidebar.")

if __name__ == "__main__":
    main()