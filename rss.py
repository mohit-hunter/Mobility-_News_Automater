from services import ScraperService
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    service = ScraperService()
    print("Starting news collector...")
    result = service.run_scraper()
    
    print("\n" + "="*60)
    print("=== SUMMARY STATISTICS ===")
    print("="*60)
    print(f"Status: {result.get('status')}")
    print(f"New articles: {result.get('new_articles')}")
    print(f"Total articles in master: {result.get('total_articles')}")
    if result.get('current_run_file'):
        print(f"Current Run File: {result.get('current_run_file')}")
    print("="*60)

if __name__ == "__main__":
    main()