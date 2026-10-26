import os
import sys
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
# Load environment variables from .env file
load_dotenv()

# Environment Variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-rag-kb")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "langgraph-agentic-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PDF_DIR = os.getenv("PDF_DIR", "pdf")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


def check_api_keys(require_all: bool = True):
    """Verify that required API keys are populated in the environment."""
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not TAVILY_API_KEY:
        missing.append("TAVILY_API_KEY")
    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")

    if missing:
        print("\n[!] Missing required API keys in .env:")
        for key in missing:
            print(f"    - {key}")
        print("\nPlease update the .env file with your valid API keys before continuing.\n")
        if require_all:
            sys.exit(1)
        return False
    return True


_embeddings = None
def get_embeddings():
    """Lazily load HuggingFace sentence transformer embeddings."""
    global _embeddings
    if _embeddings is None:
        
        print("[Config] Initializing HuggingFace embeddings (all-MiniLM-L6-v2)...")
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


_llm = None
def get_llm():
    """Lazily initialize ChatGroq model."""
    global _llm
    if _llm is None:
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set in .env")
        
        _llm = ChatGroq(
            model=GROQ_MODEL,
            temperature=0,
        )
    return _llm


_tavily = None
def get_tavily_search():
    """Lazily initialize Tavily search tool."""
    global _tavily
    if _tavily is None:
        if not TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY is not set in .env")
        
        _tavily = TavilySearch(
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _tavily


_retriever = None
def get_retriever():
    """Lazily initialize Pinecone vectorstore retriever."""
    global _retriever
    if _retriever is None:
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is not set in .env")

        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(PINECONE_INDEX_NAME)
        vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=PINECONE_NAMESPACE,
        )
        _retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 4,
                "namespace": PINECONE_NAMESPACE,
            }
        )
    return _retriever
