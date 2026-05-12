from requests_html import HTMLSession

url = "https://wikifarmer.com/library/en/article/mango-quality-standards-and-export-insights"

session = HTMLSession()
response = session.get(url)

# Render JavaScript (this will download Chromium on first run)
response.html.render(timeout=20)

content = []

for tag in response.html.find('h1, h2, h3, p, li'):
    text = tag.text.strip()
    if text:
        content.append(text)

print(f"Found {len(content)} content items")
print("\nFirst 20 items:")
for i, item in enumerate(content[:20], 1):
    print(f"{i}. {item[:100]}...")

session.close()
