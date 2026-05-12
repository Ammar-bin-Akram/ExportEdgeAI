import re
import os
from pathlib import Path

def load_documents_as_text(data_path):
    """Load all markdown files as complete text (not chunked)"""
    documents = []
    data_dir = Path(data_path)
    
    for md_file in data_dir.glob("*.md"):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Create a simple document object
            doc_obj = {
                'content': content,
                'filename': md_file.name,
                'filepath': str(md_file)
            }
            documents.append(doc_obj)
            
        except Exception as e:
            print(f"Error loading {md_file}: {e}")
    
    return documents

def clean_document(text: str) -> str:
    """
    Clean mango-export Markdown documents for RAG:
    - Remove irrelevant sections (TOC, preface, team, bibliography)
    - Remove repeated headers
    - Normalize whitespace and text
    - Lowercase letters
    """
    
    # 1. Remove Table of Contents - handles both formats
    text = re.sub(
        r'^\s*#\s*(table of contents|contents)\s*$[\s\S]*?(?=^\s*#\s+)',
        '',
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )

    # 2. Remove Preface / Introduction / Acknowledgements / Research Team
    text = re.sub(r"#\s*Preface\s*\n(?:.*\n)*?(?=#|\n[IVX]+\.)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"#\s*Introduction\s*\n(?:.*\n)*?(?=#|\n[IVX]+\.)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"#\s*Acknowledgements\s*\n(?:.*\n)*?(?=#|\n[IVX]+\.)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"#\s*Research Team\s*\n(?:.*\n)*?(?=#|\n[IVX]+\.)", "", text, flags=re.IGNORECASE)

    # 3. Remove Bibliography / Literature Cited (everything from heading to end)
    text = re.sub(r"#\s*Literature Cited.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"#\s*Bibliography.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"\n[IVX]+\.\s*Literature Cited.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    
    # 4. Remove APPENDIX sections
    text = re.sub(r"APPENDIX\s+[IVX]+.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    
    # 5. Disclaimers / USDA legal statements
    text = re.sub(r"The U\.S\. Department of Agriculture.*?employer\.", "", text, flags=re.DOTALL | re.IGNORECASE)
    
    # 6. Remove repeated main headers (like "Rule for the Importation...")
    text = re.sub(r"(#\s*Rule for the Importation of Mangoes.*\n)+", "", text, flags=re.IGNORECASE)
    
    # 7. Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)  # max 2 newlines
    text = re.sub(r"[ \t]+", " ", text)     # replace multiple spaces/tabs with single space
    
    # 8. Remove image tags and placeholders
    text = re.sub(r"\[image removed\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # Remove markdown images
    text = re.sub(r"<img.*?>", "", text, flags=re.IGNORECASE)  # Remove HTML images
    
    # 9. Remove consecutive punctuation
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"-{2,}", "-", text)
    
    # 10. Strip leading/trailing spaces from lines
    text = "\n".join(line.strip() for line in text.splitlines())
    
    # 11. Lowercase all letters for consistency
    text = text.lower()
    
    return text.strip()

def save_cleaned_documents(documents, output_path):
    """Save cleaned documents to files"""
    os.makedirs(output_path, exist_ok=True)
    
    for doc in documents:
        filename = os.path.join(output_path, f"cleaned_{doc['filename']}")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(doc['content'])