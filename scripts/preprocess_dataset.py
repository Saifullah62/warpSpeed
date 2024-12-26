import os
import json
import PyPDF2
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchPaper:
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: str
    version: str
    category: str
    subcategory: Optional[str]
    pdf_path: str
    original_path: str
    processed_date: str
    hash: Optional[str] = None
    extracted_text: Optional[str] = None
    metadata: Optional[Dict] = None

class DatasetPreprocessor:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.papers: List[ResearchPaper] = []
        self.metadata_dir = self.base_dir / "metadata"
        self.processed_dir = self.base_dir / "processed_data"
        self.metadata_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        
        # Load existing metadata if available
        self.existing_metadata = self._load_existing_metadata()

    def _load_existing_metadata(self) -> Dict:
        """Load existing metadata to avoid reprocessing."""
        try:
            metadata_file = self.metadata_dir / "papers_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return {paper['original_path']: paper for paper in json.load(f)}
            return {}
        except Exception as e:
            logger.error(f"Error loading existing metadata: {e}")
            return {}

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate file hash for tracking changes."""
        import hashlib
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def extract_arxiv_info(self, pdf_path: str) -> Dict:
        """Extract arXiv ID and version from PDF filename."""
        filename = Path(pdf_path).stem
        arxiv_id = filename.split('v')[0]
        version = f"v{filename.split('v')[1]}" if 'v' in filename else "v1"
        return {"arxiv_id": arxiv_id, "version": version}

    def extract_pdf_text(self, pdf_path: str) -> Optional[str]:
        """Extract text content from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return None

    def process_paper(self, pdf_path: str, category: str, subcategory: Optional[str] = None):
        """Process a single research paper."""
        try:
            original_path = str(pdf_path)
            file_hash = self._calculate_file_hash(pdf_path)
            
            # Check if file was already processed and hasn't changed
            if original_path in self.existing_metadata:
                existing_paper = self.existing_metadata[original_path]
                if existing_paper.get('hash') == file_hash:
                    logger.info(f"Skipping already processed file: {pdf_path}")
                    self.papers.append(ResearchPaper(**existing_paper))
                    return

            arxiv_info = self.extract_arxiv_info(pdf_path)
            text = self.extract_pdf_text(pdf_path)
            
            # Create paper object with basic info
            paper = ResearchPaper(
                title="",  # To be filled by metadata
                authors=[],  # To be filled by metadata
                abstract="",  # To be filled by metadata
                arxiv_id=arxiv_info["arxiv_id"],
                version=arxiv_info["version"],
                category=category,
                subcategory=subcategory,
                pdf_path=str(pdf_path),
                original_path=original_path,
                processed_date=datetime.now().isoformat(),
                hash=file_hash,
                extracted_text=text
            )
            self.papers.append(paper)
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")

    def process_directory(self):
        """Process all PDF files in the dataset directory."""
        start_time = datetime.now()
        logger.info(f"Starting preprocessing at {start_time}")
        
        data_dir = self.base_dir / "data"
        if not data_dir.exists():
            logger.error(f"Data directory not found: {data_dir}")
            return
        
        # Process all PDF files in the data directory
        for pdf_file in data_dir.glob("**/*.pdf"):
            # Determine category from parent directory
            category = pdf_file.parent.name if pdf_file.parent.name != "data" else "uncategorized"
            logger.info(f"Processing file in category {category}: {pdf_file.name}")
            self.process_paper(str(pdf_file), category)
        
        end_time = datetime.now()
        logger.info(f"Preprocessing completed at {end_time}")
        logger.info(f"Total processing time: {end_time - start_time}")

    def save_metadata(self):
        """Save processed metadata to JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save individual paper metadata
        papers_metadata = [asdict(paper) for paper in self.papers]
        metadata_file = self.metadata_dir / "papers_metadata.json"
        metadata_backup = self.metadata_dir / f"papers_metadata_{timestamp}.json"
        
        # Backup existing metadata if it exists
        if metadata_file.exists():
            shutil.copy2(metadata_file, metadata_backup)
        
        # Save new metadata
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(papers_metadata, f, indent=2)

        # Save dataset statistics
        stats = {
            "timestamp": timestamp,
            "total_papers": len(self.papers),
            "papers_by_category": {},
            "versions_distribution": {},
            "processing_summary": {
                "new_papers": len([p for p in self.papers if p.original_path not in self.existing_metadata]),
                "updated_papers": len([p for p in self.papers if p.original_path in self.existing_metadata]),
                "total_processed": len(self.papers)
            }
        }
        
        for paper in self.papers:
            # Count papers by category
            stats["papers_by_category"][paper.category] = \
                stats["papers_by_category"].get(paper.category, 0) + 1
            
            # Count versions
            stats["versions_distribution"][paper.version] = \
                stats["versions_distribution"].get(paper.version, 0) + 1

        with open(self.metadata_dir / f"dataset_stats_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        # Save a summary report
        report = f"""Dataset Processing Report
Timestamp: {timestamp}
Total Papers: {stats['total_papers']}
New Papers: {stats['processing_summary']['new_papers']}
Updated Papers: {stats['processing_summary']['updated_papers']}

Papers by Category:
{json.dumps(stats['papers_by_category'], indent=2)}

Version Distribution:
{json.dumps(stats['versions_distribution'], indent=2)}
"""
        with open(self.metadata_dir / f"processing_report_{timestamp}.txt", 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Metadata saved with timestamp {timestamp}")
        logger.info(f"Total papers: {stats['total_papers']}")
        logger.info(f"New papers: {stats['processing_summary']['new_papers']}")
        logger.info(f"Updated papers: {stats['processing_summary']['updated_papers']}")

def main():
    base_dir = "c:/Users/bryan/BRYANDEVELOPMENT/STAR TREK TECH"
    preprocessor = DatasetPreprocessor(base_dir)
    
    logger.info("Starting dataset preprocessing...")
    preprocessor.process_directory()
    
    logger.info("Saving metadata...")
    preprocessor.save_metadata()
    
    logger.info("Preprocessing completed!")

if __name__ == "__main__":
    main()
