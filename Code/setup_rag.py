#!/usr/bin/env python3
"""
Quick setup and test script for RAG pipeline with LLM integration
Usage: python setup_rag.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_api_key():
    """Check if Google API key is set"""
    # First try to import from config (which loads .env)
    try:
        from Code.language.data_config import GOOGLE_API_KEY
        api_key = GOOGLE_API_KEY
    except:
        # Fallback to direct environment variable
        api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        print("❌ Google API key not found!")
        print("\n🔧 Setup Instructions:")
        print("1. Get API key from https://aistudio.google.com/")
        print("2. Add to .env file: GOOGLE_API_KEY=your_api_key_here")
        print("   OR set environment variable:")
        print("   Windows: set GOOGLE_API_KEY=your_api_key_here")
        print("   Linux/Mac: export GOOGLE_API_KEY=your_api_key_here")
        print("\n📖 See RAG_SETUP.md for detailed instructions")
        return False
    else:
        print(f"✅ Google API key found: {api_key[:10]}...{api_key[-4:]}")
        return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'google.generativeai',
        'langchain',
        'langchain_community',
        'faiss'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def test_llm_only():
    """Test LLM without full pipeline"""
    try:
        from Code.language.llm_manager import LLMManager
        
        print("\n🤖 Testing LLM connection...")
        llm = LLMManager()
        
        test_prompt = "What is a mango?"
        print(f"Test query: {test_prompt}")
        
        response = llm.query_llm(test_prompt)
        print(f"✅ Response: {response[:100]}...")
        
        return True
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return False

def run_quick_setup():
    """Run quick setup and testing"""
    print("🚀 RAG Pipeline Quick Setup")
    print("=" * 50)
    
    # Check API key
    print("\n1. Checking Google API key...")
    if not check_api_key():
        return False
    
    # Check dependencies
    print("\n2. Checking dependencies...")
    if not check_dependencies():
        return False
    
    # Test LLM connection
    print("\n3. Testing LLM connection...")
    if not test_llm_only():
        return False
    
    print("\n✅ Setup complete! Ready to run RAG pipeline.")
    print("\n🏃‍♂️ Next steps:")
    print("   python Code/data_loader.py  # Run full pipeline")
    print("   OR")
    print("   python -c \"from Code.data_loader import load_existing_pipeline; load_existing_pipeline()\"")
    
    return True

def quick_rag_test():
    """Run a quick RAG test with existing data"""
    try:
        from Code.language.data_loader import load_existing_pipeline
        from Code.language.llm_manager import LLMManager
        
        print("\n🔍 Loading existing pipeline...")
        vector_store, retriever, embeddings = load_existing_pipeline()
        
        print("🤖 Initializing LLM...")
        llm = LLMManager()
        
        test_query = "What are the temperature requirements for mango storage?"
        print(f"\n❓ Test query: {test_query}")
        
        result = llm.rag_query(test_query, retriever)
        
        if result["status"] == "success":
            print(f"\n💬 Answer: {result['answer']}")
            print(f"\n📊 Used {result['num_retrieved']} sources")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ Quick RAG test failed: {e}")
        print("💡 Try running the full pipeline first: python Code/data_loader.py")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        quick_rag_test()
    else:
        run_quick_setup()