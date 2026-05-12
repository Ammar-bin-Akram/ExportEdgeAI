import re


import re

def get_text(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    match = re.search(
        r'^\s*#\s*(table of contents|contents)\s*$',
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )

    if match:
        print("✅ TOC heading found at position:", match.start())
    else:
        print("❌ TOC heading not found")

    return text


def extract_toc(text: str) -> str | None:
    match = re.search(
        r'^\s*#\s*(Table of Contents|Contents)\s*$[\s\S]*?(?=^\s*#\s+)',
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )

    return match.group(0) if match else None


def remove_toc(text: str) -> str:
    return re.sub(
        r'^\s*#\s*(table of contents|contents)\s*$[\s\S]*?(?=^\s*#\s+)',
        '',
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )


# text = get_text('data_for_llm\APHIS-2006-0121-0010_content.md')
text = get_text('data_for_llm\Mango_Inspection_Instructions[1]USA.md')
toc = extract_toc(text)
if toc:
    print("\nExtracted TOC:\n", toc)
    cleaned_text = remove_toc(text)

    with open("cleaned_data\\debug_cleaned.md", "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print("TOC removed and written to debug_cleaned.md")
else:
    print("\nNo TOC found.")




