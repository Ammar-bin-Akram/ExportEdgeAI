from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import time

url = "https://wikifarmer.com/library/en/article/mango-quality-standards-and-export-insights"

# Set up Chrome options
chrome_options = Options()
# chrome_options.add_argument('--headless')  # Run in background
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--disable-blink-features=AutomationControlled')
chrome_options.add_argument('--disable-logging')  # Reduce log output
chrome_options.add_argument('--log-level=3')  # Suppress errors
chrome_options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
chrome_options.add_experimental_option('useAutomationExtension', False)
chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36')

# Initialize the driver
driver = webdriver.Chrome(options=chrome_options)
driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

try:
    driver.get(url)
    
    # Wait for page to load
    time.sleep(5)
    
    # Try to close cookie consent if it appears
    try:
        # Wait for and click "Allow all" or similar button
        cookie_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Allow all') or contains(text(), 'Accept')]"))
        )
        cookie_button.click()
        time.sleep(2)
    except:
        # If no cookie button found, just continue
        pass
    
    # Wait for main content to load
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.TAG_NAME, "article"))
    )
    
    # Give more time for dynamic content
    time.sleep(3)
    
    # Get the page source after JavaScript has rendered
    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    # Find the main article content (avoid cookie banners, navigation, etc.)
    article = soup.find('article') or soup.find('main') or soup
    
    content = []
    
    for tag in article.find_all(['h1','h2','h3','p','li']):
        text = tag.get_text(strip=True)
        # Filter out short/irrelevant text and common cookie/UI elements
        if text and len(text) > 15 and not any(word in text.lower() for word in ['cookie', 'consent', 'necessary', 'preferences', 'statistics']):
            # Add formatting based on tag type
            if tag.name == 'h1':
                content.append(f"# {text}")
            elif tag.name == 'h2':
                content.append(f"## {text}")
            elif tag.name == 'h3':
                content.append(f"### {text}")
            elif tag.name == 'li':
                content.append(f"• {text}")
            else:
                content.append(text)
    
    print(f"Found {len(content)} content items")
    print("\nFirst 20 items:")
    for i, item in enumerate(content[:20], 1):
        print(f"{i}. {item[:100]}...")  # Print first 100 chars of each
    
    # Save to file with better formatting
    with open('mango_content.txt', 'w', encoding='utf-8') as f:
        for item in content:
            f.write(item + '\n\n')
    print(f"\n✓ Content saved to mango_content.txt")
        
finally:
    driver.quit()

