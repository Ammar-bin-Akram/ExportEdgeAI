from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def extract_markdown_sections(text):
    """
    Extract sections based on Markdown headings
    """
    sections = []
    lines = text.split('\n')
    current_section = {'heading': '', 'content': '', 'level': 0}
    
    for line in lines:
        # Check for markdown headings
        if line.strip().startswith('#'):
            # Save previous section if it has content
            if current_section['content'].strip():
                sections.append(current_section)
            
            # Start new section
            level = len(line) - len(line.lstrip('#'))
            heading = line.strip('#').strip()
            current_section = {
                'heading': heading,
                'content': line + '\n',
                'level': level
            }
        else:
            current_section['content'] += line + '\n'
    
    # Add the last section
    if current_section['content'].strip():
        sections.append(current_section)
    
    return sections

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Hierarchical chunking: First by Markdown headings, then by size if needed
    """
    # Fallback text splitter for large sections
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    global_chunk_id = 0
    
    for doc in documents:
        # Extract sections based on headings
        sections = extract_markdown_sections(doc['content'])
        
        for section in sections:
            section_content = section['content'].strip()
            
            if not section_content:
                continue
            
            # If section is small enough, keep as one chunk
            if len(section_content) <= chunk_size:
                chunk_metadata = {
                    'source': doc['filename'],
                    'filepath': doc['filepath'],
                    'chunk_id': global_chunk_id,
                    'section_heading': section['heading'],
                    'heading_level': section['level'],
                    'chunk_type': 'section',
                    'chunk_size': len(section_content)
                }
                
                chunk_doc = Document(
                    page_content=section_content,
                    metadata=chunk_metadata
                )
                chunks.append(chunk_doc)
                global_chunk_id += 1
            
            else:
                # Split large sections while preserving context
                section_chunks = text_splitter.split_text(section_content)
                
                for i, chunk_text in enumerate(section_chunks):
                    # Add section heading context to each chunk
                    if section['heading'] and not chunk_text.startswith('#'):
                        chunk_with_context = f"# {section['heading']}\n\n{chunk_text}"
                    else:
                        chunk_with_context = chunk_text
                    
                    chunk_metadata = {
                        'source': doc['filename'],
                        'filepath': doc['filepath'],
                        'chunk_id': global_chunk_id,
                        'section_heading': section['heading'],
                        'heading_level': section['level'],
                        'chunk_type': 'subsection',
                        'subsection_index': i,
                        'total_subsections': len(section_chunks),
                        'chunk_size': len(chunk_with_context)
                    }
                    
                    chunk_doc = Document(
                        page_content=chunk_with_context,
                        metadata=chunk_metadata
                    )
                    chunks.append(chunk_doc)
                    global_chunk_id += 1
    
    return chunks