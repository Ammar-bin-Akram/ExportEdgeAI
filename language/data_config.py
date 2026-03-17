# Configuration file for data loading pipeline
import os
import logging
from pathlib import Path

# Setup basic logging for config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent.parent / '.env'
    load_dotenv(env_path)
    if os.getenv('GOOGLE_API_KEY'):
        logger.info(" Environment variables loaded successfully")
    else:
        logger.warning("  GOOGLE_API_KEY not found in .env file")
except ImportError:
    logger.warning("Warning: python-dotenv not installed. Install with: pip install python-dotenv")
    logger.warning("Falling back to system environment variables...")

# Directory paths
# Get the directory where this config file is located (language folder)
CONFIG_DIR = Path(__file__).parent

# Data source (relative to project root)
DATA_PATH = str(Path(__file__).parent.parent.parent / "data_for_llm")

# Output directories (all in language folder)
OUTPUT_PATH = str(CONFIG_DIR / "cleaned_data")
CHUNKS_PATH = str(CONFIG_DIR / "chunks")
VECTOR_STORE_PATH = str(CONFIG_DIR / "vector_store")
RAG_RESULTS_PATH = str(CONFIG_DIR)  # RAG results will be saved in language folder

# Chunking parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Embedding model configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = "cpu"

# Retriever configuration
SEARCH_TYPE = "similarity"
RETRIEVAL_K = 5

# LLM Configuration
# LM Studio local server (OpenAI-compatible API)
LM_STUDIO_BASE_URL = os.getenv('LM_STUDIO_BASE_URL', 'http://192.168.0.101:1234/v1')
LLM_MODEL_NAME = "deepseek-r1-distill-qwen-1.5B"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 1024

# RAG Configuration
RETRIEVAL_CONTEXT_LIMIT = 2  # Number of chunks to retrieve for context
CONTEXT_WINDOW_SIZE = 2000  # Max characters for context (smaller for local models)

# Test queries for retriever testing
TEST_QUERIES = [
    "What are the requirements for mango inspection?",
    "How should mangoes be stored during transport?",
    "What defects are checked during mango quality inspection?",
    "What are the temperature requirements for mango storage?",
    "What are the sampling procedures for mango inspection?"
]

# RAG-specific test queries (more complex questions for LLM testing)
RAG_TEST_QUERIES = [
    "What are the temperature requirements for mango storage?",
    "What defects should be checked during mango inspection?",
    "What are the sampling procedures for mango inspection?",
    "How should mangoes be prepared for export?",
    "What are the quality standards for exported mangoes?",
    "What are the proper procedures for handling mangoes during inspection?",
    "What documentation is required for mango export?",
    "How do you determine if mangoes are ready for harvest and export?"
]