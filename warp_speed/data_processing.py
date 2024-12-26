"""
Data processing utilities for the Warp Speed Dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Union, Optional
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Process research papers and extract relevant information."""
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the data processor.
        
        Args:
            data_dir: Directory containing the raw data files
        """
        self.data_dir = Path(data_dir)
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def process_papers(self, papers: List[Dict]) -> List[Dict]:
        """
        Process a list of papers and extract relevant information.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            List of processed paper dictionaries
        """
        processed_papers = []
        for paper in papers:
            try:
                processed = self._process_single_paper(paper)
                if processed:
                    processed_papers.append(processed)
            except Exception as e:
                logger.error(f"Error processing paper {paper.get('id', 'unknown')}: {e}")
        
        return processed_papers
    
    def _process_single_paper(self, paper: Dict) -> Optional[Dict]:
        """
        Process a single paper.
        
        Args:
            paper: Paper dictionary
            
        Returns:
            Processed paper dictionary or None if processing fails
        """
        try:
            # Extract basic information
            processed = {
                'id': paper['id'],
                'title': paper['title'],
                'authors': paper['authors'],
                'abstract': paper['abstract'],
                'category': paper['category'],
                'version': paper['version']
            }
            
            # Extract and clean content
            if 'content' in paper:
                processed['content'] = self._clean_content(paper['content'])
            
            # Extract references
            if 'references' in paper:
                processed['references'] = self._process_references(paper['references'])
            
            # Extract metadata
            processed['metadata'] = self._extract_metadata(paper)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error in _process_single_paper: {e}")
            return None
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize paper content."""
        # Remove special characters
        content = content.replace('\x00', '')
        
        # Normalize whitespace
        content = ' '.join(content.split())
        
        return content
    
    def _process_references(self, references: List) -> List[str]:
        """Process and validate paper references."""
        valid_refs = []
        for ref in references:
            if isinstance(ref, str) and ref.strip():
                valid_refs.append(ref.strip())
        return valid_refs
    
    def _extract_metadata(self, paper: Dict) -> Dict:
        """Extract additional metadata from paper."""
        metadata = {}
        
        # Extract publication date
        if 'publication_date' in paper:
            metadata['publication_date'] = paper['publication_date']
        
        # Extract keywords
        if 'keywords' in paper:
            metadata['keywords'] = paper['keywords']
        
        # Extract citations
        if 'citations' in paper:
            metadata['citations'] = paper['citations']
        
        return metadata
    
    def save_processed_data(self, processed_papers: List[Dict], output_file: Union[str, Path]):
        """
        Save processed papers to file.
        
        Args:
            processed_papers: List of processed paper dictionaries
            output_file: Path to output file
        """
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame for easier saving
        df = pd.DataFrame(processed_papers)
        
        # Save to parquet format for efficiency
        df.to_parquet(output_file)
        logger.info(f"Saved {len(processed_papers)} processed papers to {output_file}")
    
    def load_processed_data(self, input_file: Union[str, Path]) -> List[Dict]:
        """
        Load processed papers from file.
        
        Args:
            input_file: Path to input file
            
        Returns:
            List of processed paper dictionaries
        """
        input_file = Path(input_file)
        if not input_file.exists():
            raise FileNotFoundError(f"Input file {input_file} not found")
        
        # Load from parquet format
        df = pd.read_parquet(input_file)
        
        # Convert to list of dictionaries
        papers = df.to_dict('records')
        logger.info(f"Loaded {len(papers)} processed papers from {input_file}")
        
        return papers
