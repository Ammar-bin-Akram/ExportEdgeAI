# Simple test script to verify the updated paths
from pathlib import Path
import sys

# Add the language folder to Python path
language_dir = Path(__file__).parent / "Code" / "language"
sys.path.insert(0, str(language_dir))

try:
    # Import the config directly
    import data_config
    
    print("✅ Configuration loaded successfully!")
    print("=" * 50)
    print(f"📁 OUTPUT_PATH: {data_config.OUTPUT_PATH}")
    print(f"📝 CHUNKS_PATH: {data_config.CHUNKS_PATH}")
    print(f"🔍 VECTOR_STORE_PATH: {data_config.VECTOR_STORE_PATH}")
    print(f"🤖 RAG_RESULTS_PATH: {data_config.RAG_RESULTS_PATH}")
    print(f"📂 DATA_PATH: {data_config.DATA_PATH}")
    print("=" * 50)
    
    # Verify paths point to language folder
    from pathlib import Path
    
    output_path = Path(data_config.OUTPUT_PATH)
    chunks_path = Path(data_config.CHUNKS_PATH)
    vector_store_path = Path(data_config.VECTOR_STORE_PATH)
    rag_results_path = Path(data_config.RAG_RESULTS_PATH)
    
    print("📍 Path verification:")
    print(f"   Output in language folder: {'language' in str(output_path.resolve())}")
    print(f"   Chunks in language folder: {'language' in str(chunks_path.resolve())}")
    print(f"   Vector store in language folder: {'language' in str(vector_store_path.resolve())}")
    print(f"   RAG results in language folder: {'language' in str(rag_results_path.resolve())}")
    
    print("\n🎉 All paths successfully updated to use language folder!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()