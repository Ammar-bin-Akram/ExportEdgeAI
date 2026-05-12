# Language processing modules for document processing pipeline
"""
This package contains modules for:
- Document loading and cleaning
- Text chunking with hierarchical structure  
- Vector store management and embeddings
- Citation management and retrieval testing
- LLM integration and RAG pipeline
- Configuration management
"""

# Modules are available for import but not pre-imported to avoid dependency issues
__all__ = [
    'document_processor',
    'chunker', 
    'vector_store_manager',
    'citation_manager',
    'llm_manager',
    'data_config'
]