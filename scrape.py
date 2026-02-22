import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import argparse

# NOTE ON ETHICAL SCRAPING:
# When scraping a public website like ikman.lk, it's important to respect their
# terms of service and robots.txt file. 
# We implement randomized delays (time.sleep) to ensure we do not overwhelm 
# their servers with requests, behaving more like a normal human user. 
# Scraping should be done responsibly.

def scrape_ikman_vehicles(output_path, max_pages=5):
    """
    Scrapes vehicle ads from ikman.lk and saves to a CSV.
    Note: ikman.lk DOM structure can change. This script uses typical class structures
    as of standard classified lists, but may need adjustment if their UI updates.
    """
    base_url = "https://ikman.lk/en/ads/sri-lanka/vehicles?sort=date&order=desc&buy_now=0&urgent=0&page="
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                      'AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/113.0.0.0 Safari/537.36'
    }

    data = []

    for page in range(1, max_pages + 1):
        print(f"Scraping page {page}...")
        url = base_url + str(page)
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching page {page}: {e}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all ad containers (this class might change over time)
        # Note: In a robust scraper, we'd use more precise or up-to-date selectors
        ads = soup.find_all('li', class_='normal--2QYVk gtm-normal-ad')
        
        if not ads:
            print(f"No ads found on page {page}. Checking alternative selectors or page limit reached.")
            # Often ikman uses wrapper components, we can fallback to standard a tags if needed
            ads = soup.find_all('a', class_='card-link--3ssYv')

        for ad in ads:
            ad_info = {}
            
            # Extract Title
            title_elem = ad.find('h2')
            if title_elem:
                ad_info['Title'] = title_elem.text.strip()
            else:
                 continue # Title is mandatory for us

            # Extract Price
            price_elem = ad.find('div', class_='price--3SnqI')
            if price_elem:
                ad_info['Price'] = price_elem.text.strip()
                
            # Extract Location and description/mileage briefly
            desc_elem = ad.find('div', class_='description--2-ez3')
            if desc_elem:
                ad_info['Description'] = desc_elem.text.strip()
                
            # Usually location and type are stored in 'description' split by comma
            # For this basic script we scrape what's available
            
            data.append(ad_info)

        # Respectful delay between requests
        delay = random.uniform(1.5, 3.5)
        print(f"Sleeping for {delay:.2f} seconds...")
        time.sleep(delay)

    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    if not df.empty:
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Successfully saved {len(df)} records to {output_path}")
    else:
        print("No data was scraped. Please check CSS selectors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape ikman.lk vehicle ads.")
    parser.add_argument("--output", type=str, default="scraped_data.csv", help="Output CSV path")
    parser.add_argument("--pages", type=int, default=3, help="Maximum number of pages to scrape")
    
    args = parser.parse_args()
    
    print(f"Starting scraper for max {args.pages} pages. Output mapping to {args.output}")
    scrape_ikman_vehicles(output_path=args.output, max_pages=args.pages)
