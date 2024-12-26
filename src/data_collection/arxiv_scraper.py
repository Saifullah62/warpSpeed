"""
arXiv Scraper for collecting research papers related to warp drive technology.
Handles pagination and full PDF downloads.
"""

import os
import time
import json
import logging
import requests
import feedparser
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urlparse, parse_qs
import backoff
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ArXivPaper:
    """Represents an arXiv research paper."""
    title: str
    authors: List[str]
    abstract: str
    arxiv_id: str
    categories: List[str]
    primary_category: str
    published_date: str
    updated_date: str
    pdf_url: str
    local_pdf_path: Optional[str]
    comment: Optional[str]
    journal_ref: Optional[str]
    doi: Optional[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the paper to a dictionary."""
        return asdict(self)

class ArXivScraper:
    """Scraper for arXiv papers with robust error handling and pagination support."""
    
    # Base URLs
    BASE_URL = "http://export.arxiv.org/api/query"
    PDF_BASE_URL = "https://arxiv.org/pdf"
    
    # arXiv categories relevant to warp drive research
    CATEGORIES = {
        'physics': [
            'gr-qc',    # General Relativity and Quantum Cosmology
            'hep-th',   # High Energy Physics - Theory
            'quant-ph', # Quantum Physics
            'physics.space-ph',  # Space Physics
            'physics.gen-ph',    # General Physics
            'physics.class-ph',  # Classical Physics
            'physics.optics',    # Optics
            'physics.plasm-ph'   # Plasma Physics
        ],
        'materials': [
            'cond-mat.mtrl-sci',  # Materials Science
            'cond-mat.str-el',    # Strongly Correlated Materials
            'cond-mat.other'      # Other Condensed Matter
        ],
        'mathematics': [
            'math-ph',  # Mathematical Physics
            'math.MP',  # Mathematical Physics
            'math.DG'   # Differential Geometry
        ]
    }
    
    def __init__(self, output_dir: str = "data/arxiv"):
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
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
    
    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, ConnectionError),
        max_tries=5
    )
    def _make_request(self, url: str, params: Dict[str, Any] = None) -> requests.Response:
        """Make a request with exponential backoff retry."""
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response
    
    def _create_search_query(self, keywords: List[str], category: str) -> str:
        """Create a search query string for arXiv API."""
        # Combine keywords with OR
        keyword_query = " OR ".join(f'"{k}"' for k in keywords)
        
        # Add category filter
        if category in self.CATEGORIES:
            category_filter = " OR ".join(f"cat:{cat}" for cat in self.CATEGORIES[category])
            return f"({keyword_query}) AND ({category_filter})"
        
        return keyword_query
    
    def _download_pdf(self, arxiv_id: str, category: str) -> Optional[str]:
        """Download PDF for a paper and return the local path."""
        pdf_dir = os.path.join(self.pdf_dir, category)
        os.makedirs(pdf_dir, exist_ok=True)
        
        local_path = os.path.join(pdf_dir, f"{arxiv_id.replace('/', '_')}.pdf")
        if os.path.exists(local_path):
            logger.info(f"PDF already exists: {local_path}")
            return local_path
        
        try:
            pdf_url = f"{self.PDF_BASE_URL}/{arxiv_id}.pdf"
            response = self._make_request(pdf_url, params={'download': 1})
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Downloaded PDF: {local_path}")
            return local_path
            
        except Exception as e:
            logger.error(f"Error downloading PDF for {arxiv_id}: {str(e)}")
            if os.path.exists(local_path):
                os.remove(local_path)
            return None
    
    def _parse_paper(self, entry: Dict[str, Any], category: str) -> ArXivPaper:
        """Parse arXiv API entry into ArXivPaper object and download PDF."""
        # Extract arXiv ID from the id field
        arxiv_id = entry.get('id', '').split('/')[-1]
        if '/' in arxiv_id:
            arxiv_id = arxiv_id.split('/')[-1]
        
        # Download PDF
        pdf_path = self._download_pdf(arxiv_id, category)
        
        # Extract authors
        authors = []
        if 'authors' in entry:
            for author in entry.authors:
                if hasattr(author, 'name'):
                    authors.append(author.name)
                elif isinstance(author, str):
                    authors.append(author)
        
        # Extract categories
        categories = []
        if 'tags' in entry:
            for tag in entry.tags:
                if hasattr(tag, 'term'):
                    categories.append(tag.term)
        
        return ArXivPaper(
            title=entry.get('title', '').strip().replace('\n', ' '),
            authors=authors,
            abstract=entry.get('summary', '').strip().replace('\n', ' '),
            arxiv_id=arxiv_id,
            categories=categories,
            primary_category=categories[0] if categories else '',
            published_date=entry.get('published', ''),
            updated_date=entry.get('updated', ''),
            pdf_url=f"{self.PDF_BASE_URL}/{arxiv_id}",
            local_pdf_path=pdf_path,
            comment=entry.get('comment', ''),
            journal_ref=entry.get('journal_ref', ''),
            doi=entry.get('doi', '')
        )
    
    def fetch_papers(
        self,
        category: str,
        keywords: List[str],
        max_results: int = 500,
        start_year: Optional[int] = None
    ) -> List[ArXivPaper]:
        """
        Fetch papers from arXiv for given category and keywords.
        
        Args:
            category: Category name from self.CATEGORIES
            keywords: List of keywords to search for
            max_results: Maximum number of results to return
            start_year: Optional start year for filtering
            
        Returns:
            List of ArXivPaper objects with downloaded PDFs
        """
        if category not in self.CATEGORIES:
            raise ValueError(f"Invalid category: {category}")
        
        all_papers = []
        batch_size = 100  # arXiv API maximum
        start = 0
        
        while len(all_papers) < max_results:
            try:
                # Create query
                query = self._create_search_query(keywords, category)
                
                # Set up parameters
                params = {
                    'search_query': query,
                    'start': start,
                    'max_results': min(batch_size, max_results - len(all_papers)),
                    'sortBy': 'lastUpdatedDate',
                    'sortOrder': 'descending'
                }
                
                # Make request
                response = self._make_request(self.BASE_URL, params)
                feed = feedparser.parse(response.content)
                
                if not feed.entries:
                    break
                
                # Process entries
                papers = []
                for entry in feed.entries:
                    try:
                        # Skip papers before start_year if specified
                        if start_year:
                            pub_year = int(entry.published[:4])
                            if pub_year < start_year:
                                continue
                        
                        paper = self._parse_paper(entry, category)
                        if paper:
                            papers.append(paper)
                    except Exception as e:
                        logger.error(f"Error parsing paper: {str(e)}")
                        continue
                
                all_papers.extend(papers)
                
                if len(papers) < batch_size:  # No more results
                    break
                
                start += batch_size
                time.sleep(3)  # Be nice to arXiv API
                
            except Exception as e:
                logger.error(f"Error fetching papers: {str(e)}")
                break
        
        logger.info(f"Found {len(all_papers)} papers")
        return all_papers[:max_results]
    
    def save_papers(self, papers: List[ArXivPaper], category: str):
        """Save papers metadata to JSON file."""
        if not papers:
            logger.warning("No papers to save")
            return
        
        # Create output directory if it doesn't exist
        category_dir = os.path.join(self.output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(category_dir, f"arxiv_papers_{timestamp}.json")
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([paper.to_dict() for paper in papers], f, indent=2)
        
        logger.info(f"Saved {len(papers)} paper metadata to {filename}")
    
    def scrape_category(
        self,
        category: str,
        keywords: List[str],
        max_results: int = 500,
        start_year: Optional[int] = None
    ):
        """Scrape papers for a category and save them."""
        try:
            papers = self.fetch_papers(category, keywords, max_results, start_year)
            self.save_papers(papers, category)
        except Exception as e:
            logger.error(f"Error scraping category {category}: {str(e)}")
            raise
    
    def scrape_all_categories(
        self,
        keywords: List[str],
        max_results: int = 500,
        start_year: Optional[int] = None
    ):
        """Scrape papers for all categories."""
        for category in self.CATEGORIES:
            try:
                logger.info(f"Processing category: {category}")
                self.scrape_category(category, keywords, max_results, start_year)
                time.sleep(5)  # Sleep between categories
            except Exception as e:
                logger.error(f"Error processing category {category}: {str(e)}")
                continue
