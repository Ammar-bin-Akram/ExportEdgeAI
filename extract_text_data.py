import os
from pathlib import Path
from mistralai import Mistral, DocumentURLChunk
from mistralai.models import OCRResponse
import re


api_key="#"
client = Mistral(api_key=api_key)
print('API connected')

def get_combined_markdown(ocr_response: OCRResponse, filename: str = None) -> str:
    # function to convert OCR data to markdown
    markdowns: list[str] = []
    for page in ocr_response.pages:
        page_markdown = page.markdown
        page_markdown = re.sub(r'!\[.*?\]\(.*?\)', '[Image removed]', page_markdown)
        markdowns.append(page_markdown)
    return "\n\n".join(markdowns)

def process_single_pdf(pdf_path: str, output_path: str = None):
    """
    Process a single PDF file and extract text using OCR.
    
    Args:
        pdf_path: Path to the PDF file (string or Path object)
        output_path: Optional path for the output markdown file. 
                     If None, creates a .md file with the same name as the PDF
    
    Returns:
        str: Path to the generated markdown file
    """
    pdf_path = Path(pdf_path)
    
    # Validate PDF exists
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Set default output path if not provided
    if output_path is None:
        output_path = pdf_path.with_suffix(".md")
    else:
        output_path = Path(output_path)
    
    print(f"Processing: {pdf_path.name}")
    print(f"Output will be saved to: {output_path}")
    
    # Upload file to Mistral API
    uploaded_file = client.files.upload(
        file={
            "file_name": pdf_path.stem,
            "content": pdf_path.read_bytes(),
        },
        purpose="ocr",
    )
    
    # Get signed URL for processing
    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)
    
    # Process OCR
    print("Running OCR...")
    pdf_response = client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url),
        model="mistral-ocr-latest",
        include_image_base64=False
    )
    
    # Convert to markdown
    combined_markdown = get_combined_markdown(pdf_response, filename=pdf_path.stem)
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(combined_markdown)
    
    print(f"✓ Successfully processed! Text saved to: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    # Example usage - modify this path to your PDF file
    pdf_files_path = "raw_data_LLM"
    pdf_file = f"{pdf_files_path}/45_Mangoes.pdf"  # Change this to your PDF path
    
    # Option 1: Use default output path (same name as PDF but .md extension)
    # process_single_pdf(pdf_file)
    
    # Option 2: Specify custom output path
    output_path = 'data_for_llm'
    process_single_pdf(pdf_file, f"{output_path}/{Path(pdf_file).stem}.md")