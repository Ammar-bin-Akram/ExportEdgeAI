import os
import json

def save_chunks_for_citation(chunks, chunks_path):
    """
    Save chunks in multiple formats for citation and debugging purposes
    """
    print("Saving chunks for citation purposes...")
    os.makedirs(chunks_path, exist_ok=True)
    
    # 1. Save as JSON for easy access
    chunks_data = []
    for i, chunk in enumerate(chunks):
        chunk_data = {
            'chunk_index': i,
            'content': chunk.page_content,
            'metadata': chunk.metadata,
            'content_preview': chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content
        }
        chunks_data.append(chunk_data)
    
    # Save all chunks as JSON
    with open(os.path.join(chunks_path, "all_chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)
    
    # 2. Save individual chunk files for easy retrieval
    for i, chunk in enumerate(chunks):
        chunk_filename = f"chunk_{i:04d}_{chunk.metadata['source']}_part_{chunk.metadata['chunk_id']}.txt"
        chunk_filepath = os.path.join(chunks_path, chunk_filename)
        
        with open(chunk_filepath, "w", encoding="utf-8") as f:
            f.write(f"Source: {chunk.metadata['source']}\n")
            f.write(f"Chunk ID: {chunk.metadata['chunk_id']}\n")
            f.write(f"Section: {chunk.metadata.get('section_heading', 'N/A')}\n")
            f.write(f"Heading Level: {chunk.metadata.get('heading_level', 'N/A')}\n")
            f.write(f"Chunk Type: {chunk.metadata.get('chunk_type', 'N/A')}\n")
            f.write(f"File Path: {chunk.metadata['filepath']}\n")
            f.write(f"Chunk Size: {chunk.metadata['chunk_size']} characters\n")
            f.write(f"Global Index: {i}\n")
            f.write("-" * 50 + "\n")
            f.write(chunk.page_content)
    
    # 3. Create a mapping file for quick chunk lookup
    chunk_mapping = {}
    for i, chunk in enumerate(chunks):
        chunk_mapping[i] = {
            'source_file': chunk.metadata['source'],
            'chunk_id': chunk.metadata['chunk_id'],
            'section_heading': chunk.metadata.get('section_heading', 'N/A'),
            'heading_level': chunk.metadata.get('heading_level', 'N/A'),
            'chunk_type': chunk.metadata.get('chunk_type', 'N/A'),
            'file_path': chunk.metadata['filepath'],
            'chunk_file': f"chunk_{i:04d}_{chunk.metadata['source']}_part_{chunk.metadata['chunk_id']}.txt"
        }
    
    with open(os.path.join(chunks_path, "chunk_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(chunk_mapping, f, indent=2, ensure_ascii=False)
    
    # 4. Create summary statistics
    stats = {
        'total_chunks': len(chunks),
        'source_files': list(set(chunk.metadata['source'] for chunk in chunks)),
        'chunks_per_file': {},
        'avg_chunk_size': sum(chunk.metadata['chunk_size'] for chunk in chunks) / len(chunks),
        'min_chunk_size': min(chunk.metadata['chunk_size'] for chunk in chunks),
        'max_chunk_size': max(chunk.metadata['chunk_size'] for chunk in chunks)
    }
    
    for chunk in chunks:
        source = chunk.metadata['source']
        if source not in stats['chunks_per_file']:
            stats['chunks_per_file'][source] = 0
        stats['chunks_per_file'][source] += 1
    
    with open(os.path.join(chunks_path, "chunk_statistics.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(chunks)} chunks to {chunks_path}")
    print(f"📊 Chunk statistics saved to chunk_statistics.json")
    print(f"🗂️  Chunk mapping saved to chunk_mapping.json")

def load_chunk_by_index(chunk_index, chunks_path):
    """
    Load a specific chunk by its global index for citation
    """
    try:
        with open(os.path.join(chunks_path, "chunk_mapping.json"), "r", encoding="utf-8") as f:
            mapping = json.load(f)
        
        if str(chunk_index) in mapping:
            chunk_info = mapping[str(chunk_index)]
            chunk_file = os.path.join(chunks_path, chunk_info['chunk_file'])
            
            with open(chunk_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            return {
                'index': chunk_index,
                'content': content,
                'metadata': chunk_info
            }
        else:
            print(f"Chunk {chunk_index} not found")
            return None
            
    except Exception as e:
        print(f"Error loading chunk {chunk_index}: {e}")
        return None

def get_citation_info(chunk_index, chunks_path):
    """
    Get citation information for a specific chunk
    """
    chunk_data = load_chunk_by_index(chunk_index, chunks_path)
    if chunk_data:
        metadata = chunk_data['metadata']
        return {
            'source_document': metadata['source_file'],
            'chunk_reference': f"Chunk {metadata['chunk_id']}",
            'file_path': metadata['file_path'],
            'citation': f"Source: {metadata['source_file']}, Chunk {metadata['chunk_id']}"
        }
    return None

def test_retriever(retriever, test_queries, chunks_path):
    """
    Test the retriever with sample queries and show chunk indices for citation
    """
    print("\n" + "="*60)
    print("TESTING RETRIEVER WITH CITATION INFO")
    print("="*60)
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 50)
        
        try:
            # Use invoke() instead of get_relevant_documents()
            results = retriever.invoke(query)
            
            for i, doc in enumerate(results[:3], 1):  # Show top 3 results
                # Find the chunk index from the vector store
                chunk_index = None
                with open(os.path.join(chunks_path, "all_chunks.json"), "r", encoding="utf-8") as f:
                    all_chunks = json.load(f)
                    
                for j, chunk_data in enumerate(all_chunks):
                    if (chunk_data['metadata']['source'] == doc.metadata['source'] and 
                        chunk_data['metadata']['chunk_id'] == doc.metadata['chunk_id']):
                        chunk_index = j
                        break
                
                print(f"\n📄 Result {i}:")
                print(f"   Source: {doc.metadata['source']}")
                print(f"   Section: {doc.metadata.get('section_heading', 'N/A')}")
                print(f"   Chunk Type: {doc.metadata.get('chunk_type', 'N/A')}")
                print(f"   Chunk ID: {doc.metadata['chunk_id']}")
                print(f"   Global Index: {chunk_index}")
                print(f"   Citation: Source: {doc.metadata['source']}, Section: {doc.metadata.get('section_heading', 'N/A')}, Chunk {doc.metadata['chunk_id']}")
                print(f"   Content: {doc.page_content[:300]}...")
                
        except Exception as e:
            print(f"❌ Error retrieving documents: {e}")