import undetected_chromedriver as uc
from bs4 import BeautifulSoup
import time

url = "https://wikifarmer.com/library/en/article/mango-quality-standards-and-export-insights"

# Use undetected chromedriver
driver = uc.Chrome(headless=False)  # Set to True once it works

try:
    driver.get(url)
    
    # Wait for page to load completely
    time.sleep(8)
    
    # Get the page source after JavaScript has rendered
    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    content = []
    
    for tag in soup.find_all(['h1','h2','h3','p','li']):
        text = tag.get_text(strip=True)
        if text and len(text) > 10:  # Filter out very short text
            content.append(text)
    
    print(f"Found {len(content)} content items")
    print("\nFirst 20 items:")
    for i, item in enumerate(content[:20], 1):
        print(f"{i}. {item[:100]}...")
        
finally:
    driver.quit()
