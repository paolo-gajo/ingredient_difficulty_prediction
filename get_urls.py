import requests
from bs4 import BeautifulSoup
import time
from tqdm.auto import tqdm

def extract_urls(base_url, max_pages):
    urls = []
    
    for page in tqdm(range(1, max_pages + 1)):
        if page == 1:
            url = base_url
        else:
            url = f"{base_url}page{page}/"
        
        print(f"Scraping: {url}")
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        
        if response.status_code != 200:
            print(f"Failed to retrieve page {page}")
            continue
        
        soup = BeautifulSoup(response.text, 'html.parser')
        recipe_links = soup.select("h2.gz-title a")
        
        for link in recipe_links:
            urls.append(link['href'])
        
        time.sleep(1)  # To avoid overloading the server
    
    return urls

if __name__ == "__main__":
    BASE_URL = "https://www.giallozafferano.it/ricette-cat/"
    MAX_PAGES = 487  # Last page
    
    extracted_urls = extract_urls(BASE_URL, MAX_PAGES)
    
    # Save to file
    with open("extracted_urls.txt", "w") as f:
        for url in extracted_urls:
            f.write(url + "\n")
    
    print(f"Extracted {len(extracted_urls)} URLs. Saved to extracted_urls.txt")
