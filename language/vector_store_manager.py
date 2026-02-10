import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def create_embeddings_and_vectorstore(chunks, vector_store_path):
    """
    Create embeddings using HuggingFace and store in FAISS vector store
    """
    print("Creating embeddings...")
    
    # Initialize embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # Create FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save vector store
    os.makedirs(vector_store_path, exist_ok=True)
    vector_store.save_local(vector_store_path)
    
    # Save chunks metadata for debugging
    with open(os.path.join(vector_store_path, "chunks_metadata.pkl"), "wb") as f:
        pickle.dump([chunk.metadata for chunk in chunks], f)
    
    print(f"Vector store created with {len(chunks)} chunks")
    return vector_store, embeddings

def load_vectorstore(vector_store_path):
    """
    Load existing vector store
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    return vector_store, embeddings

def create_retriever(vector_store, search_type="similarity", k=5):
    """
    Create a retriever from the vector store
    """
    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k}
    )
    return retriever