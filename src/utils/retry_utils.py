import requests
from typing import Dict, Any, Optional, List, Callable, TypeVar
import logging
from bs4 import BeautifulSoup
import time
import random
from functools import wraps
from requests.exceptions import RequestException

logger = logging.getLogger(__name__)

T = TypeVar('T')

def with_retry(max_attempts: int = 3, base_delay: float = 1.0) -> Callable:
    """Decorator for retrying functions with exponential backoff."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

def handle_http_error(response: requests.Response) -> None:
    """Handle HTTP errors with appropriate logging and actions."""
    if response.status_code != 200:
        error_msg = f"HTTP {response.status_code}: {response.reason}"
        logger.error(error_msg)
        response.raise_for_status()

def safe_get_text(element: Optional[BeautifulSoup], default: str = "") -> str:
    """Safely extract text from a BeautifulSoup element."""
    return element.get_text(strip=True) if element else default

def safe_get_attr(element: Optional[BeautifulSoup], attr: str, default: str = "") -> str:
    """Safely get attribute from a BeautifulSoup element."""
    return element.get(attr, default) if element else default

def validate_response(response: Optional[requests.Response], url: str = None) -> bool:
    """Validate response content and structure.
    
    Args:
        response: The HTTP response to validate
        url: Optional URL for error logging
        
    Returns:
        bool: True if response is valid, False otherwise
    """
    if not response:
        logger.error(f"No response object from {url if url else 'unknown URL'}")
        return False
        
    try:
        if not response.content:
            logger.error(f"Empty response content from {url if url else 'unknown URL'}")
            return False
        
        content_type = response.headers.get('content-type', '')
        if 'html' in content_type.lower():
            soup = BeautifulSoup(response.content, 'html.parser')
            if not soup.find():
                logger.error(f"Invalid HTML structure from {url if url else 'unknown URL'}")
                return False
        elif 'json' in content_type.lower():
            try:
                response.json()
            except ValueError:
                logger.error(f"Invalid JSON response from {url if url else 'unknown URL'}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating response from {url if url else 'unknown URL'}: {str(e)}")
        return False

class DataValidator:
    """Validates scraped data for completeness and quality."""
    
    REQUIRED_FIELDS = {
        'title': str,
        'authors': list,
        'publication_date': str,
        'full_text': str,
        'doi': str,
        'references': list,
        'methodology': str,
        'results': str,
        'data_tables': list,
        'figures': list
    }
    
    OPTIONAL_FIELDS = {
        'abstract': str,
        'keywords': list,
        'institution': str,
        'funding': list,
        'supplementary_materials': list,
        'code_repository': str,
        'dataset_url': str
    }
    
    @classmethod
    def validate_record(cls, record: Dict[str, Any]) -> tuple[bool, List[str]]:
        """
        Validate a single record for completeness.
        Returns (is_valid, list of missing required fields)
        """
        missing_fields = []
        
        # Check required fields
        for field, expected_type in cls.REQUIRED_FIELDS.items():
            if field not in record:
                missing_fields.append(field)
            elif not isinstance(record[field], expected_type):
                missing_fields.append(f"{field} (wrong type)")
            elif expected_type == str and not record[field].strip():
                missing_fields.append(f"{field} (empty)")
            elif expected_type == list and not record[field]:
                missing_fields.append(f"{field} (empty)")
        
        # Specific validation for full_text
        if 'full_text' in record and len(record['full_text']) < 1000:  # Minimum length for full text
            missing_fields.append("full_text (too short)")
        
        return len(missing_fields) == 0, missing_fields
    
    @classmethod
    def enrich_record(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        """Add optional fields with None if they don't exist."""
        for field in cls.OPTIONAL_FIELDS:
            if field not in record:
                record[field] = None
        return record

class SiteValidator:
    """Validates site availability and structure before scraping."""
    
    def __init__(self, url: str, required_elements: Dict[str, str]):
        self.url = url
        self.required_elements = required_elements
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def check_availability(self, max_retries: int = 3) -> bool:
        """Check if the site is available with retries."""
        for attempt in range(max_retries):
            try:
                response = self.session.get(self.url, timeout=30)
                if response.status_code == 200:
                    return True
                
                logger.warning(
                    f"Site {self.url} returned status code {response.status_code} "
                    f"on attempt {attempt + 1}/{max_retries}"
                )
                
                # Add delay between retries
                time.sleep(random.uniform(2, 5))
                
            except Exception as e:
                logger.error(f"Error checking {self.url}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(random.uniform(5, 10))
                
        return False
    
    def validate_structure(self) -> tuple[bool, List[str]]:
        """
        Validate that all required elements are present.
        Returns (is_valid, list of missing elements)
        """
        try:
            response = self.session.get(self.url, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            missing_elements = []
            for element, selector in self.required_elements.items():
                if not soup.select(selector):
                    missing_elements.append(element)
            
            return len(missing_elements) == 0, missing_elements
            
        except Exception as e:
            logger.error(f"Error validating structure for {self.url}: {str(e)}")
            return False, ["Failed to validate structure"]
    
    def validate_content_access(self, test_article_url: Optional[str] = None) -> bool:
        """Validate that we can access full article content."""
        try:
            if test_article_url:
                response = self.session.get(test_article_url, timeout=30)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Check for common full-text indicators
                content_indicators = [
                    "full-text",
                    "article-body",
                    "main-content",
                    "paper-content",
                    "download-pdf"
                ]
                
                for indicator in content_indicators:
                    if soup.find(class_=lambda x: x and indicator in x.lower()):
                        return True
                
                return False
            
            return True  # Skip validation if no test URL provided
            
        except Exception as e:
            logger.error(f"Error validating content access: {str(e)}")
            return False

class ScraperRetryPolicy:
    """Defines retry policy for failed scrapers."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 30.0):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.attempt_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def should_retry(self, scraper_name: str) -> bool:
        """Determine if a scraper should be retried based on its history."""
        if scraper_name not in self.attempt_history:
            return True
        
        attempts = self.attempt_history[scraper_name]
        if len(attempts) >= self.max_attempts:
            return False
        
        # Check if last attempt was successful
        if attempts and attempts[-1].get('success', False):
            return False
        
        return True
    
    def get_delay(self, scraper_name: str) -> float:
        """Calculate delay before next retry with exponential backoff."""
        attempts = len(self.attempt_history.get(scraper_name, []))
        return self.base_delay * (2 ** attempts) + random.uniform(0, self.base_delay)
    
    def record_attempt(self, scraper_name: str, success: bool, error: str = None) -> None:
        """Record an attempt for a scraper."""
        if scraper_name not in self.attempt_history:
            self.attempt_history[scraper_name] = []
        
        attempt = {
            'timestamp': time.time(),
            'success': success,
            'error': error
        }
        
        self.attempt_history[scraper_name].append(attempt)
    
    def get_attempt_count(self, scraper_name: str) -> int:
        """Get the number of attempts for a scraper."""
        return len(self.attempt_history.get(scraper_name, []))

def validate_sites_before_scraping() -> bool:
    """Validate all sites before starting the scraping process."""
    sites = {
        "Harvard Dataverse": {
            "url": "https://dataverse.harvard.edu",
            "required_elements": {
                "search": "input#searchbox",
                "results": "div.results-container"
            }
        },
        "ORNL": {
            "url": "https://www.ornl.gov",
            "required_elements": {
                "search": "input#search-box",
                "navigation": "nav.main-navigation"
            }
        },
        "Perimeter Institute": {
            "url": "https://www.perimeterinstitute.ca",
            "required_elements": {
                "search": "input.search-field",
                "content": "div.main-content"
            }
        }
    }
    
    all_valid = True
    for site_name, config in sites.items():
        validator = SiteValidator(config["url"], config["required_elements"])
        if not validator.check_availability():
            logger.error(f"Site {site_name} is not available")
            all_valid = False
            continue
            
        logger.info(f"Site {site_name} is available")
    
    return all_valid
