# Main data loader pipeline - enhanced modular version
import os
import sys
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from document_processor import load_documents_as_text, clean_document, save_cleaned_documents
from chunker import chunk_documents
from vector_store_manager import create_embeddings_and_vectorstore, load_vectorstore, create_retriever
from citation_manager import save_chunks_for_citation, test_retriever
from llm_manager import test_llm_integration, save_rag_results
from data_config import (DATA_PATH, OUTPUT_PATH, CHUNKS_PATH, VECTOR_STORE_PATH, RAG_RESULTS_PATH,
                        CHUNK_SIZE, CHUNK_OVERLAP, SEARCH_TYPE, RETRIEVAL_K, TEST_QUERIES)

# Setup logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def validate_config() -> bool:
    """Validate configuration and paths"""
    logger.info("Validating configuration...")
    
    # Check if source data exists
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data path not found: {DATA_PATH}")
        return False
    
    # Create necessary directories
    for path in [OUTPUT_PATH, CHUNKS_PATH, VECTOR_STORE_PATH]:
        os.makedirs(path, exist_ok=True)
    
    logger.info("✅ Configuration validated")
    return True

def process_documents(skip_if_exists: bool = False) -> List[Dict[str, Any]]:
    """Process documents: load, clean, and save"""
    start_time = time.time()
    logger.info("Step 1: Loading and cleaning documents...")
    
    # Check if cleaned documents already exist
    if skip_if_exists and os.path.exists(OUTPUT_PATH) and os.listdir(OUTPUT_PATH):
        logger.info(f"Cleaned documents found in {OUTPUT_PATH}, skipping processing")
        # Load existing cleaned documents (simplified - you might want to implement this)
        docs = load_documents_as_text(DATA_PATH)
        elapsed = time.time() - start_time
        logger.info(f"✅ Loaded {len(docs)} existing documents in {elapsed:.2f}s")
        return docs
    
    docs = load_documents_as_text(DATA_PATH)
    logger.info(f"Loaded {len(docs)} documents from {DATA_PATH}")
    
    # Clean documents with progress tracking
    cleaned_docs = []
    for i, doc in enumerate(docs, 1):
        cleaned_text = clean_document(doc['content'])
        doc['content'] = cleaned_text
        cleaned_docs.append(doc)
        if i % 5 == 0 or i == len(docs):  # Progress every 5 docs or at end
            logger.info(f"Cleaned {i}/{len(docs)} documents")
    
    # Save cleaned documents
    save_cleaned_documents(cleaned_docs, OUTPUT_PATH)
    elapsed = time.time() - start_time
    logger.info(f"✅ Step 1 completed: {len(cleaned_docs)} documents processed in {elapsed:.2f}s")
    
    return cleaned_docs

def create_chunks(documents: List[Dict[str, Any]]) -> List[Any]:
    """Create chunks from documents"""
    start_time = time.time()
    logger.info("Step 2: Chunking documents...")
    
    chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    elapsed = time.time() - start_time
    
    # Log chunking statistics
    total_chars = sum(len(doc['content']) for doc in documents)
    avg_chunk_size = total_chars / len(chunks) if chunks else 0
    
    logger.info(f"✅ Step 2 completed: Created {len(chunks)} chunks")
    logger.info(f"   Average chunk size: {avg_chunk_size:.0f} characters")
    logger.info(f"   Processing time: {elapsed:.2f}s")
    
    return chunks

def create_vector_store(chunks: List[Any], force_recreate: bool = False) -> Tuple[Any, Any]:
    """Create embeddings and vector store"""
    start_time = time.time()
    logger.info("Step 3: Creating embeddings and vector store...")
    
    # Check if vector store exists and we don't want to force recreate
    vector_store_exists = os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss"))
    
    if vector_store_exists and not force_recreate:
        logger.info("Existing vector store found, loading...")
        vector_store, embeddings = load_vectorstore(VECTOR_STORE_PATH)
        elapsed = time.time() - start_time
        logger.info(f"✅ Step 3 completed: Vector store loaded in {elapsed:.2f}s")
    else:
        logger.info(f"Creating new vector store with {len(chunks)} chunks...")
        vector_store, embeddings = create_embeddings_and_vectorstore(chunks, VECTOR_STORE_PATH)
        elapsed = time.time() - start_time
        logger.info(f"✅ Step 3 completed: Vector store created in {elapsed:.2f}s")
    
    return vector_store, embeddings

def setup_retriever(vector_store: Any) -> Any:
    """Setup retriever from vector store"""
    start_time = time.time()
    logger.info("Step 4: Creating retriever...")
    
    retriever = create_retriever(vector_store, search_type=SEARCH_TYPE, k=RETRIEVAL_K)
    elapsed = time.time() - start_time
    
    logger.info(f"✅ Step 4 completed: Retriever configured (k={RETRIEVAL_K}, {elapsed:.2f}s)")
    return retriever

def test_pipeline(retriever: Any, run_tests: bool = True) -> None:
    """Test the retriever with predefined queries"""
    if not run_tests:
        logger.info("Skipping retriever tests")
        return
        
    start_time = time.time()
    logger.info("Step 5: Testing retriever...")
    
    test_retriever(retriever, TEST_QUERIES, CHUNKS_PATH)
    elapsed = time.time() - start_time
    
    logger.info(f"✅ Step 5 completed: Retriever tested with {len(TEST_QUERIES)} queries in {elapsed:.2f}s")

def test_rag_pipeline(retriever: Any, run_tests: bool = True, custom_queries: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Test the complete RAG pipeline with LLM integration"""
    if not run_tests:
        logger.info("Skipping RAG pipeline tests")
        return []
        
    start_time = time.time()
    logger.info("Step 6: Testing RAG pipeline with LLM...")
    
    try:
        results = test_llm_integration(retriever, custom_queries)
        elapsed = time.time() - start_time
        
        if results:
            # Create timestamped filename in language folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(RAG_RESULTS_PATH, f"rag_test_results_{timestamp}.json")
            save_rag_results(results, filename)
            
            # Log statistics
            successful_queries = sum(1 for r in results if r.get('status') == 'success')
            avg_response_time = elapsed / len(results) if results else 0
            
            logger.info(f"✅ Step 6 completed: RAG pipeline tested successfully")
            logger.info(f"   Queries processed: {len(results)}")
            logger.info(f"   Success rate: {successful_queries}/{len(results)} ({100*successful_queries/len(results):.1f}%)")
            logger.info(f"   Average response time: {avg_response_time:.2f}s per query")
            logger.info(f"   Results saved to: {filename}")
            return results
        else:
            logger.error("RAG pipeline testing failed - no results generated")
            return []
    except Exception as e:
        logger.error(f"RAG pipeline testing error: {e}")
        return []

def main(skip_existing: bool = False, force_recreate_vectors: bool = False, 
         run_tests: bool = True, custom_queries: Optional[List[str]] = None) -> Tuple[Any, Any, Any]:
    """
    Main pipeline: Load -> Clean -> Chunk -> Save -> Embed -> Retrieve -> Test
    
    Args:
        skip_existing: Skip processing if cleaned documents exist
        force_recreate_vectors: Force recreation of vector store
        run_tests: Whether to run retriever and RAG tests
        custom_queries: Custom queries for RAG testing
    """
    pipeline_start = time.time()
    logger.info("=" * 60)
    logger.info("🚀 Starting Enhanced Document Processing Pipeline")
    logger.info("=" * 60)
    
    # Validate configuration
    if not validate_config():
        raise ValueError("Configuration validation failed")
    
    try:
        # Process documents
        cleaned_docs = process_documents(skip_if_exists=skip_existing)
        
        # Create chunks
        chunks = create_chunks(cleaned_docs)
        
        # Save chunks for citation
        chunk_start = time.time()
        logger.info("Step 3: Saving chunks for citation...")
        save_chunks_for_citation(chunks, CHUNKS_PATH)
        chunk_elapsed = time.time() - chunk_start
        logger.info(f"✅ Step 3 completed: Chunks saved in {chunk_elapsed:.2f}s")
        
        # Create vector store
        vector_store, embeddings = create_vector_store(chunks, force_recreate=force_recreate_vectors)
        
        # Setup retriever
        retriever = setup_retriever(vector_store)
        
        # Test pipeline
        test_pipeline(retriever, run_tests=run_tests)
        
        # Test RAG with LLM
        rag_results = test_rag_pipeline(retriever, run_tests=run_tests, custom_queries=custom_queries)
        
        # Final summary
        total_elapsed = time.time() - pipeline_start
        logger.info("=" * 60)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📁 Cleaned documents: {OUTPUT_PATH}")
        logger.info(f"📝 Chunks: {CHUNKS_PATH} ({len(chunks)} chunks)")
        logger.info(f"🔍 Vector store: {VECTOR_STORE_PATH}")
        if rag_results:
            logger.info(f"🤖 RAG test results: {len(rag_results)} queries processed")
        logger.info(f"⏱️  Total pipeline time: {total_elapsed:.2f}s")
        logger.info("=" * 60)
        
        return vector_store, retriever, embeddings
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

def load_existing_pipeline(run_tests: bool = False) -> Tuple[Any, Any, Any]:
    """Load existing vector store and create retriever"""
    try:
        logger.info("Loading existing pipeline...")
        start_time = time.time()
        
        vector_store, embeddings = load_vectorstore(VECTOR_STORE_PATH)
        retriever = create_retriever(vector_store, search_type=SEARCH_TYPE, k=RETRIEVAL_K)
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Existing pipeline loaded successfully in {elapsed:.2f}s")
        
        if run_tests:
            test_pipeline(retriever, run_tests=True)
            test_rag_pipeline(retriever, run_tests=True)
        
        return vector_store, retriever, embeddings
    except Exception as e:
        logger.warning(f"Could not load existing pipeline: {e}")
        logger.info("Creating new pipeline...")
        return main()

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Enhanced Document Processing Pipeline for RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Run full pipeline
  %(prog)s --load-existing              # Load existing pipeline
  %(prog)s --skip-existing              # Skip if processed docs exist
  %(prog)s --force-recreate-vectors     # Force vector store recreation
  %(prog)s --no-tests                   # Skip all tests
  %(prog)s --log-level DEBUG            # Enable debug logging
        """
    )
    
    parser.add_argument('--load-existing', action='store_true',
                       help='Load existing pipeline instead of creating new')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip document processing if cleaned docs exist')
    parser.add_argument('--force-recreate-vectors', action='store_true',
                       help='Force recreation of vector store')
    parser.add_argument('--no-tests', action='store_true',
                       help='Skip retriever and RAG tests')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set logging level')
    parser.add_argument('--custom-queries', nargs='+',
                       help='Custom queries for RAG testing')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Setup logging with specified level
    logger = setup_logging(args.log_level)
    
    try:
        if args.load_existing:
            vector_store, retriever, embeddings = load_existing_pipeline(run_tests=not args.no_tests)
        else:
            vector_store, retriever, embeddings = main(
                skip_existing=args.skip_existing,
                force_recreate_vectors=args.force_recreate_vectors,
                run_tests=not args.no_tests,
                custom_queries=args.custom_queries
            )
        
        logger.info("Pipeline ready for use!")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        raise

