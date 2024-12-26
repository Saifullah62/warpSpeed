import os
import logging
from src.scrapers.ornl_scraper import ORNLScraper
from src.scrapers.perimeter_scraper import PerimeterScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def test_ornl_scraper():
    """Test ORNL scraper functionality"""
    logger.info("Testing ORNL Scraper...")
    output_dir = "test_data/ornl"
    os.makedirs(output_dir, exist_ok=True)
    
    scraper = ORNLScraper(output_dir)
    
    # Test quantum research
    logger.info("Testing fetch_quantum_research...")
    quantum_data = scraper.fetch_quantum_research()
    logger.info(f"Quantum research records: {len(quantum_data)}")
    for i, record in enumerate(quantum_data[:3], 1):
        logger.info(f"Record {i}:")
        logger.info(f"  Title: {record.get('title', 'N/A')}")
        logger.info(f"  Authors: {record.get('authors', [])}")
    
    # Test facility data
    logger.info("\nTesting fetch_facility_data...")
    facility_data = scraper.fetch_facility_data()
    logger.info(f"Facility records: {len(facility_data)}")
    for i, record in enumerate(facility_data[:3], 1):
        logger.info(f"Record {i}:")
        logger.info(f"  Name: {record.get('name', 'N/A')}")
        logger.info(f"  Description: {record.get('description', 'N/A')[:100]}...")

def test_perimeter_scraper():
    """Test Perimeter Institute scraper functionality"""
    logger.info("\nTesting Perimeter Institute Scraper...")
    output_dir = "test_data/perimeter"
    os.makedirs(output_dir, exist_ok=True)
    
    scraper = PerimeterScraper(output_dir)
    
    # Test quantum gravity research
    logger.info("Testing fetch_quantum_gravity_research...")
    quantum_data = scraper.fetch_quantum_gravity_research()
    logger.info(f"Quantum gravity records: {len(quantum_data)}")
    for i, record in enumerate(quantum_data[:3], 1):
        logger.info(f"Record {i}:")
        logger.info(f"  Title: {record.get('title', 'N/A')}")
        logger.info(f"  Authors: {record.get('authors', [])}")
    
    # Test research areas
    logger.info("\nTesting fetch_research_areas...")
    areas_data = scraper.fetch_research_areas()
    logger.info(f"Research area records: {len(areas_data)}")
    for i, record in enumerate(areas_data[:3], 1):
        logger.info(f"Record {i}:")
        logger.info(f"  Name: {record.get('name', 'N/A')}")
        logger.info(f"  Description: {record.get('description', 'N/A')[:100]}...")

if __name__ == "__main__":
    try:
        test_ornl_scraper()
        test_perimeter_scraper()
        logger.info("\nAll tests completed!")
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
