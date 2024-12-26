from src.data_collection.nist_scraper import NISTScraper
import os
import json

def main():
    # Create data directory if it doesn't exist
    data_dir = "data/nist"
    os.makedirs(data_dir, exist_ok=True)
    
    scraper = NISTScraper()
    print("Fetching NIST constants...")
    
    # Get and save all constants with their details
    scraper.save_constants_to_file(f"{data_dir}/nist_constants.json")

if __name__ == "__main__":
    main()
