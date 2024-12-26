"""
NASA Technical Reports Server (NTRS) Scraper.
Fetches full technical reports and research papers from NASA's public database.
"""

import requests
import json
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import backoff  # For exponential backoff
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NASADocument:
    """Represents a NASA technical document."""
    title: str
    authors: List[str]
    abstract: str
    publication_date: str
    document_id: str
    keywords: List[str]
    subject_categories: List[str]
    pdf_url: Optional[str]
    local_pdf_path: Optional[str]
    document_type: str
    source: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the document to a dictionary."""
        return asdict(self)

class NASAScraper:
    """Scraper for NASA Technical Reports Server with robust error handling."""
    
    # Base URLs
    BASE_URL = "https://ntrs.nasa.gov/api/citations/search"
    PDF_BASE_URL = "https://ntrs.nasa.gov/api/citations/"
    
    # Research categories relevant to warp drive
    CATEGORIES = {
        "propulsion": [
            "PROPULSION AND POWER",
            "ADVANCED PROPULSION",
            "SPACECRAFT PROPULSION AND POWER"
        ],
        "physics": [
            "PHYSICS",
            "THEORETICAL PHYSICS",
            "QUANTUM PHYSICS",
            "PLASMA PHYSICS"
        ],
        "materials": [
            "MATERIALS AND MANUFACTURING",
            "ADVANCED MATERIALS",
            "COMPOSITE MATERIALS"
        ],
        "space_science": [
            "SPACE SCIENCES",
            "ASTROPHYSICS",
            "COSMOLOGY"
        ]
    }
    
    def __init__(self, output_dir: str = "data/nasa"):
        """Initialize the scraper with output directory and configure requests session."""
        self.output_dir = output_dir
        self.pdf_dir = os.path.join(output_dir, "pdfs")
        os.makedirs(self.pdf_dir, exist_ok=True)
        
        # Configure session with retries
        self.session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "POST"]
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, ConnectionError),
        max_tries=5
    )
    def _make_request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make a request with exponential backoff retry."""
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response
    
    def _create_search_payload(
        self,
        category: str,
        keywords: Optional[List[str]] = None,
        start_year: Optional[int] = None,
        page: int = 1,
        page_size: int = 25
    ) -> Dict[str, Any]:
        """Create the search payload for the NTRS API."""
        
        # Build category filter
        category_filter = {
            "terms": {
                "subject_categories": self.CATEGORIES.get(category, [category])
            }
        }
        
        # Build keyword filter if provided
        keyword_filter = None
        if keywords:
            keyword_filter = {
                "multi_match": {
                    "query": " OR ".join(keywords),
                    "fields": ["abstract", "title", "keywords"]
                }
            }
        
        # Build date filter if provided
        date_filter = None
        if start_year:
            date_filter = {
                "range": {
                    "publication_date": {
                        "gte": f"{start_year}-01-01",
                        "lte": "2024-12-31"
                    }
                }
            }
        
        # Combine all filters
        must_filters = [f for f in [category_filter, keyword_filter, date_filter] if f]
        
        return {
            "query": {
                "bool": {
                    "must": must_filters
                }
            },
            "size": page_size,
            "from": (page - 1) * page_size,
            "sort": [{"publication_date": "desc"}]
        }
    
    def _create_text_from_metadata(self, data: Dict[str, Any], doc_dir: str, document_id: str) -> Optional[str]:
        """
        Create a text file from available metadata when no document is available.
        
        Args:
            data: Document metadata
            doc_dir: Directory to save the text file
            document_id: NASA document ID
            
        Returns:
            Path to created text file or None if creation failed
        """
        try:
            # Create a formatted text document from metadata
            content = []
            content.append("=" * 80)
            content.append(f"NASA Technical Document {document_id}")
            content.append("=" * 80 + "\n")
            
            # Title
            if title := data.get('title'):
                content.append(f"Title: {title}\n")
            
            # Authors
            if authors := data.get('authors', []):
                content.append("Authors:")
                for author in authors:
                    content.append(f"- {author}")
                content.append("")
            
            # Publication Info
            if date := data.get('publication_date'):
                content.append(f"Publication Date: {date}")
            if center := data.get('center'):
                content.append(f"NASA Center: {center}")
            content.append("")
            
            # Keywords and Categories
            if keywords := data.get('keywords', []):
                content.append("Keywords:")
                content.append(", ".join(keywords))
                content.append("")
            if categories := data.get('subject_categories', []):
                content.append("Subject Categories:")
                for category in categories:
                    content.append(f"- {category}")
                content.append("")
            
            # Abstract
            if abstract := data.get('abstract'):
                content.append("Abstract:")
                content.append("-" * 40)
                content.append(abstract)
                content.append("-" * 40 + "\n")
            
            # Technical Details
            if details := data.get('technical_details'):
                content.append("Technical Details:")
                content.append("-" * 40)
                content.append(details)
                content.append("-" * 40 + "\n")
            
            # Description
            if description := data.get('description'):
                content.append("Description:")
                content.append("-" * 40)
                content.append(description)
                content.append("-" * 40 + "\n")
            
            # Save as text file
            text_path = os.path.join(doc_dir, f"{document_id}.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(content))
            
            logger.info(f"Created text file from metadata: {text_path}")
            return text_path
            
        except Exception as e:
            logger.error(f"Error creating text file for {document_id}: {str(e)}")
            return None

    def _download_document(self, document_id: str, category: str) -> Optional[str]:
        """
        Download document for a NASA ID, trying multiple formats (PDF, TXT, etc).
        
        Args:
            document_id: NASA document ID
            category: Category for organizing documents
            
        Returns:
            Path to downloaded document or None if download failed
        """
        doc_dir = os.path.join(self.pdf_dir, category)
        os.makedirs(doc_dir, exist_ok=True)
        
        try:
            # Get document metadata to find available downloads
            response = self._make_request("GET", f"{self.PDF_BASE_URL}{document_id}")
            data = response.json()
            
            # Find all available downloads
            downloads = data.get('downloads', [])
            
            # Try formats in order of preference
            format_priority = ['PDF', 'TXT', 'DOC', 'HTML']
            
            for format_type in format_priority:
                format_links = [d for d in downloads if d.get('type') == format_type]
                if not format_links:
                    continue
                    
                doc_url = format_links[0]['links'].get(format_type.lower())
                if not doc_url:
                    continue
                    
                local_path = os.path.join(doc_dir, f"{document_id}.{format_type.lower()}")
                
                if os.path.exists(local_path):
                    logger.info(f"Document already exists: {local_path}")
                    return local_path
                
                try:
                    # Download document with streaming
                    response = self._make_request("GET", doc_url, stream=True)
                    
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                
                    logger.info(f"Downloaded {format_type} document: {local_path}")
                    return local_path
                    
                except Exception as e:
                    logger.error(f"Error downloading {format_type} for {document_id}: {str(e)}")
                    if os.path.exists(local_path):
                        os.remove(local_path)  # Clean up partial download
                    continue
            
            # If we get here, no formats were available
            logger.warning(f"No downloadable document formats available for {document_id}")
            
            # Try to create a text file from metadata
            if text_path := self._create_text_from_metadata(data, doc_dir, document_id):
                return text_path
            
            # Save raw metadata as JSON as final fallback
            meta_path = os.path.join(doc_dir, f"{document_id}_metadata.json")
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved metadata to {meta_path}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error accessing document {document_id}: {str(e)}")
            return None

    def _parse_document(self, doc_data: Dict[str, Any], category: str) -> NASADocument:
        """Parse NASA document data into NASADocument object and download document."""
        doc_id = doc_data['id']
        doc_path = self._download_document(doc_id, category)
        
        # Get document type from file extension or fallback to metadata
        doc_type = "metadata_only"
        if doc_path:
            ext = os.path.splitext(doc_path)[1].lower()
            if ext:
                doc_type = ext[1:]  # Remove the dot
        
        return NASADocument(
            title=doc_data.get('title', '').strip(),
            authors=doc_data.get('authors', []),
            abstract=doc_data.get('abstract', '').strip(),
            publication_date=doc_data.get('publication_date', ''),
            document_id=doc_id,
            keywords=doc_data.get('keywords', []),
            subject_categories=doc_data.get('subject_categories', []),
            pdf_url=f"{self.PDF_BASE_URL}{doc_id}/downloads/pdf",
            local_pdf_path=doc_path,
            document_type=doc_type,
            source=doc_data.get('source', '')
        )
    
    def fetch_documents(
        self,
        category: str,
        keywords: Optional[List[str]] = None,
        max_results: int = 500,
        start_year: Optional[int] = None
    ) -> List[NASADocument]:
        """
        Fetch documents from NTRS for a given category and keywords.
        
        Args:
            category: Category name from self.CATEGORIES
            keywords: Optional list of keywords to search for
            max_results: Maximum number of results to return
            start_year: Start year for date filter
            
        Returns:
            List of NASADocument objects with downloaded documents
        """
        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category: {category}")
            
        all_docs = []
        page = 1
        page_size = min(25, max_results)  # NTRS API limit is 25 per page
        
        while len(all_docs) < max_results:
            try:
                payload = self._create_search_payload(
                    category,
                    keywords,
                    start_year,
                    page,
                    page_size
                )
                
                response = self._make_request("POST", self.BASE_URL, json=payload)
                data = response.json()
                
                if not data.get('results'):
                    break
                    
                docs = []
                for doc in data['results']:
                    try:
                        parsed_doc = self._parse_document(doc, category)
                        if parsed_doc:
                            docs.append(parsed_doc)
                    except Exception as e:
                        logger.error(f"Error parsing document: {str(e)}")
                        continue
                
                all_docs.extend(docs)
                
                if len(docs) < page_size:  # No more results
                    break
                    
                page += 1
                time.sleep(2)  # Be nice to NASA's servers
                
            except Exception as e:
                logger.error(f"Error fetching documents: {str(e)}")
                break
                
        logger.info(f"Found {len(all_docs)} documents")
        return all_docs[:max_results]
    
    def save_documents(self, documents: List[NASADocument], category: str):
        """Save documents metadata to JSON file."""
        if not documents:
            logger.warning("No documents to save")
            return
            
        # Create output directory if it doesn't exist
        category_dir = os.path.join(self.output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(category_dir, f"nasa_docs_{timestamp}.json")
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([doc.to_dict() for doc in documents], f, indent=2)
        
        logger.info(f"Saved {len(documents)} document metadata to {filename}")
        
    def scrape_category(
        self,
        category: str,
        keywords: Optional[List[str]] = None,
        max_results: int = 500,
        start_year: Optional[int] = None
    ):
        """Scrape documents for a category and save them."""
        try:
            documents = self.fetch_documents(category, keywords, max_results, start_year)
            self.save_documents(documents, category)
        except Exception as e:
            logger.error(f"Error scraping category {category}: {str(e)}")
            raise
            
    def scrape_all_categories(
        self,
        keywords: Optional[List[str]] = None,
        max_results: int = 500,
        start_year: Optional[int] = None
    ):
        """Scrape documents for all categories."""
        for category in self.CATEGORIES:
            try:
                logger.info(f"Processing category: {category}")
                self.scrape_category(category, keywords, max_results, start_year)
                time.sleep(3)  # Sleep between categories
            except Exception as e:
                logger.error(f"Error processing category {category}: {str(e)}")
                continue
