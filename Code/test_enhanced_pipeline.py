# Test script for the enhanced pipeline
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from Code.language.data_loader import main, load_existing_pipeline

def test_basic_pipeline():
    """Test the basic pipeline functionality"""
    print("🧪 Testing basic pipeline...")
    
    # Test with skip existing (faster for testing)
    vector_store, retriever, embeddings = main(
        skip_existing=True,
        run_tests=True
    )
    
    print("✅ Basic pipeline test completed!")
    return vector_store, retriever, embeddings

def test_custom_queries():
    """Test with custom queries"""
    print("\n🧪 Testing with custom queries...")
    
    custom_queries = [
        "What are the main quality indicators for mango export?",
        "How do storage conditions affect mango quality?"
    ]
    
    # Load existing pipeline and test with custom queries
    vector_store, retriever, embeddings = load_existing_pipeline(run_tests=False)
    
    # Import the test function directly
    from Code.language.data_loader import test_rag_pipeline
    
    results = test_rag_pipeline(
        retriever, 
        run_tests=True,
        custom_queries=custom_queries
    )
    
    print(f"✅ Custom query test completed! Processed {len(results)} queries")
    return results

if __name__ == "__main__":
    print("🚀 Testing Enhanced RAG Pipeline")
    print("=" * 50)
    
    # Test basic functionality
    vector_store, retriever, embeddings = test_basic_pipeline()
    
    # Test custom queries
    results = test_custom_queries()
    
    print("\n" + "=" * 50)
    print("🎉 All tests completed successfully!")
    print("Your enhanced pipeline is ready for use!")