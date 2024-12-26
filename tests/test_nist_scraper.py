import unittest
from src.data_collection.nist_scraper import NISTScraper
import os

class TestNISTScraper(unittest.TestCase):
    def setUp(self):
        self.scraper = NISTScraper()
        
    def test_get_constants(self):
        """Test fetching constants list."""
        constants = self.scraper.get_constants()
        self.assertIsInstance(constants, list)
        if constants:  # Only test if we got data back
            self.assertIsInstance(constants[0], dict)
            
    def test_get_constant_details(self):
        """Test fetching details for a specific constant."""
        # Using Planck constant as an example
        details = self.scraper.get_constant_details("h")  # 'h' is the symbol for Planck constant
        if details:  # Only test if we got data back
            self.assertIsInstance(details, dict)
            self.assertIn("value", details)
            
    def test_save_constants(self):
        """Test saving constants to file."""
        test_file = "test_constants.json"
        self.scraper.save_constants_to_file(test_file)
        
        # Check if file exists and has content
        self.assertTrue(os.path.exists(test_file))
        self.assertGreater(os.path.getsize(test_file), 0)
        
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)
        
if __name__ == '__main__':
    unittest.main()
