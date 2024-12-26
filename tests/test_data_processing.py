"""
Tests for data processing functionality.
"""

import unittest
from pathlib import Path
import tempfile
import shutil
import json
from warp_speed.data_processing import DataProcessor

class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.data_processor = DataProcessor(self.test_dir)
        
        # Sample test data
        self.test_paper = {
            'id': 'test123',
            'title': 'Test Paper',
            'authors': ['Author One', 'Author Two'],
            'abstract': 'This is a test abstract.',
            'content': 'This is the main content.',
            'category': 'physics',
            'version': 'v1',
            'references': ['ref1', 'ref2'],
            'publication_date': '2024-01-01'
        }
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_process_single_paper(self):
        """Test processing of a single paper."""
        processed = self.data_processor._process_single_paper(self.test_paper)
        
        self.assertIsNotNone(processed)
        self.assertEqual(processed['id'], 'test123')
        self.assertEqual(processed['title'], 'Test Paper')
        self.assertEqual(len(processed['authors']), 2)
        self.assertIn('metadata', processed)
    
    def test_clean_content(self):
        """Test content cleaning."""
        dirty_content = "This is \x00 some dirty\n\n content  with   spaces"
        clean_content = self.data_processor._clean_content(dirty_content)
        
        self.assertNotIn('\x00', clean_content)
        self.assertEqual(clean_content, "This is some dirty content with spaces")
    
    def test_process_references(self):
        """Test reference processing."""
        refs = ['  ref1 ', '', 'ref2  ', None]
        processed_refs = self.data_processor._process_references(refs)
        
        self.assertEqual(len(processed_refs), 2)
        self.assertEqual(processed_refs[0], 'ref1')
        self.assertEqual(processed_refs[1], 'ref2')
    
    def test_extract_metadata(self):
        """Test metadata extraction."""
        metadata = self.data_processor._extract_metadata(self.test_paper)
        
        self.assertIn('publication_date', metadata)
        self.assertEqual(metadata['publication_date'], '2024-01-01')
    
    def test_save_and_load_processed_data(self):
        """Test saving and loading processed data."""
        # Process some papers
        processed_papers = self.data_processor.process_papers([self.test_paper])
        
        # Save to file
        output_file = self.test_dir / 'processed_papers.parquet'
        self.data_processor.save_processed_data(processed_papers, output_file)
        
        # Load from file
        loaded_papers = self.data_processor.load_processed_data(output_file)
        
        self.assertEqual(len(loaded_papers), len(processed_papers))
        self.assertEqual(loaded_papers[0]['id'], processed_papers[0]['id'])
    
    def test_error_handling(self):
        """Test error handling for invalid input."""
        # Test with invalid paper
        invalid_paper = {'id': 'test456'}  # Missing required fields
        processed = self.data_processor._process_single_paper(invalid_paper)
        self.assertIsNone(processed)
        
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            self.data_processor.load_processed_data(self.test_dir / 'nonexistent.parquet')

if __name__ == '__main__':
    unittest.main()
