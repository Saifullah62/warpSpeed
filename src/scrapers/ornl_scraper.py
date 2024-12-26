import requests
import pandas as pd
from typing import Dict, Any, List, Optional
from src.scrapers.base_scraper import BaseScraper
from bs4 import BeautifulSoup
import time
from src.utils.retry_utils import (
    with_retry, handle_http_error, safe_get_text,
    safe_get_attr, validate_response
)
from requests.exceptions import RequestException
import logging

class ORNLScraper(BaseScraper):
    def __init__(self, output_dir: str):
        """Initialize the ORNL scraper"""
        super().__init__(output_dir)
        self.base_url = "https://www.ornl.gov"
        self.logger = logging.getLogger("ORNLScraper")
        self.backup_urls = {
            'quantum': [
                '/research-areas/quantum-science',
                '/research-areas/quantum-computing',
                '/research-areas/quantum-information',
                '/research-areas/quantum-materials',
                '/news/quantum'
            ],
            'facilities': [
                '/research-areas/facilities',
                '/research-areas/user-facilities',
                '/research-areas/neutron-facilities',
                '/research-areas/computing-facilities'
            ]
        }
        self.session = requests.Session()
        
    def __del__(self):
        """Clean up session on deletion."""
        if hasattr(self, 'session'):
            self.session.close()

    @with_retry(max_attempts=3, base_delay=2.0)
    def _make_request(self, url: str) -> requests.Response:
        """Make HTTP request with retry logic."""
        try:
            response = self.session.get(url, timeout=30)
            handle_http_error(response)
            validate_response(response, url)
            return response
        except RequestException as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            raise

    def _try_urls(self, url_list: List[str]) -> Optional[requests.Response]:
        """Try multiple URLs until one works"""
        for url in url_list:
            try:
                full_url = f"{self.base_url}{url}"
                response = self._make_request(full_url)
                if response.status_code == 200:
                    return response
            except Exception as e:
                self.logger.debug(f"Failed to fetch {full_url}: {str(e)}")
        return None

    def _extract_research_data(self, element: BeautifulSoup) -> Dict[str, Any]:
        """Safely extract research data from HTML element."""
        try:
            return {
                'title': safe_get_text(element.find('h2', class_='title')) or 
                        safe_get_text(element.find('h3', class_='title')),
                'researchers': [
                    author.text.strip() 
                    for author in element.find_all('div', class_='author') +
                    element.find_all('span', class_='author')
                ],
                'abstract': safe_get_text(element.find('div', class_='summary')) or
                           safe_get_text(element.find('div', class_='description')),
                'facility': safe_get_text(element.find('div', class_='facility')) or
                           safe_get_text(element.find('span', class_='facility')),
                'publication_date': safe_get_text(element.find('div', class_='date')) or
                                  safe_get_text(element.find('span', class_='date')),
                'url': self.base_url + safe_get_attr(element.find('a', class_='read-more'), 'href', '')
            }
        except Exception as e:
            self.logger.error(f"Error extracting research data: {str(e)}")
            return {}

    def fetch_quantum_research(self) -> List[Dict[str, Any]]:
        """Fetch quantum research papers from ORNL website"""
        research_data = []
        
        try:
            response = self._try_urls(self.backup_urls['quantum'])
            if not response:
                self.logger.error("Failed to fetch quantum research data from any URL")
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try multiple possible container classes
            articles = (
                soup.find_all('div', class_='research-highlight') or
                soup.find_all('div', class_='news-item') or
                soup.find_all('article') or
                soup.find_all('div', class_='content-item')
            )
            
            for article in articles:
                # Look for quantum-related content
                title = safe_get_text(article, ['h2', 'h3', 'h4']).lower()
                if any(kw in title for kw in ['quantum', 'qubits', 'superposition']):
                    data = {
                        'title': safe_get_text(article, ['h2', 'h3', 'h4']),
                        'summary': safe_get_text(article, ['div.summary', 'div.description', 'p']),
                        'date': safe_get_text(article, ['div.date', 'time', 'span.date']),
                        'url': self.base_url + safe_get_attr(article, 'a', 'href')
                    }
                    if data['title']:
                        research_data.append(data)
            
            return research_data
            
        except Exception as e:
            self.logger.error(f"Error fetching quantum research: {str(e)}")
            return []

    def fetch_facility_data(self) -> List[Dict[str, Any]]:
        """Fetch facility research data from ORNL website"""
        facility_data = []
        
        try:
            response = self._try_urls(self.backup_urls['facilities'])
            if not response:
                self.logger.error("Failed to fetch facility data from any URL")
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try multiple possible container classes
            facilities = (
                soup.find_all('div', class_='facility') or
                soup.find_all('div', class_='research-facility') or
                soup.find_all('article', class_='facility') or
                soup.find_all('div', class_='content-item')
            )
            
            for facility in facilities:
                data = {
                    'name': safe_get_text(facility, ['h2', 'h3', 'h4']),
                    'description': safe_get_text(facility, ['div.description', 'div.summary', 'p']),
                    'capabilities': [
                        cap.text.strip() 
                        for cap in facility.find_all(['li', 'div'], class_=['capability', 'feature'])
                    ],
                    'url': self.base_url + safe_get_attr(facility, 'a', 'href')
                }
                if data['name']:
                    facility_data.append(data)
            
            return facility_data
            
        except Exception as e:
            self.logger.error(f"Error fetching facility data: {str(e)}")
            return []

    def fetch_research_details(self, url: str) -> Dict[str, Any]:
        """Fetch detailed information for a research project"""
        try:
            response = self._make_request(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract funding information if available
            funding = safe_get_text(soup, 'div.funding-info')
            
            # Extract related publications
            publications = [
                pub.text.strip()
                for pub in soup.find_all('div', class_='related-publication')
            ]
            
            # Extract equipment/facilities used
            equipment = [
                eq.text.strip()
                for eq in soup.find_all('span', class_='equipment-item')
            ]
            
            return {
                'funding_info': funding,
                'related_publications': publications,
                'equipment_used': equipment
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching research details from {url}: {str(e)}")
            return {}

    def fetch_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all relevant ORNL data"""
        self.logger.info("Fetching ORNL data")
        
        data = {
            'quantum_research': self.fetch_quantum_research(),
            'facility_experiments': self.fetch_facility_data()
        }
        
        # Enrich with detailed information
        for category in data:
            for i, item in enumerate(data[category]):
                if 'url' in item and item['url']:
                    details = self.fetch_research_details(item['url'])
                    data[category][i].update(details)
                    time.sleep(1)
        
        return data
            
    def preprocess_data(self, data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess ORNL data"""
        self.logger.info("Preprocessing ORNL data")
        
        try:
            # Process quantum research data
            quantum_df = pd.DataFrame(data['quantum_research'])
            if not quantum_df.empty:
                quantum_df['research_type'] = 'quantum'
                
            # Process facility experiment data
            facility_df = pd.DataFrame(data['facility_experiments'])
            if not facility_df.empty:
                facility_df['research_type'] = 'facility'
            
            # Combine datasets
            df = pd.concat([quantum_df, facility_df], ignore_index=True)
            
            # Convert dates safely
            if 'publication_date' in df.columns:
                df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
            
            # Add metadata
            df['source'] = 'ORNL'
            df['scrape_timestamp'] = pd.Timestamp.now()
            
            # Categorize research areas
            area_keywords = {
                'quantum_computing': [
                    'quantum computing', 'qubit', 'quantum circuit',
                    'quantum algorithm'
                ],
                'quantum_materials': [
                    'quantum material', 'topological', 'superconductor',
                    'quantum phase'
                ],
                'neutron_science': [
                    'neutron scattering', 'neutron diffraction',
                    'spallation'
                ],
                'particle_physics': [
                    'particle physics', 'high-energy', 'nuclear physics',
                    'isotope'
                ]
            }
            
            def categorize_research(text):
                if pd.isna(text):
                    return []
                text = str(text).lower()
                areas = []
                for area, keywords in area_keywords.items():
                    if any(kw in text for kw in keywords):
                        areas.append(area)
                return areas
            
            if 'abstract' in df.columns:
                df['research_areas'] = df['abstract'].apply(categorize_research)
            
            # Calculate research complexity score safely
            if 'equipment_used' in df.columns:
                df['complexity_score'] = df['equipment_used'].apply(
                    lambda x: len(x) if isinstance(x, list) else 0
                )
            
            self.logger.info("Preprocessing completed")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in preprocess_data: {str(e)}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(
                columns=['title', 'researchers', 'abstract', 'research_type', 
                        'source', 'scrape_timestamp']
            )

    def _scrape_quantum(self) -> pd.DataFrame:
        """Method disabled"""
        return pd.DataFrame()

    def _scrape_facilities(self) -> pd.DataFrame:
        """Method disabled"""
        return pd.DataFrame()

    def scrape(self) -> Dict[str, Any]:
        """
        ORNL scraping temporarily disabled
        """
        self.logger.info("ORNL scraping is currently disabled - skipping")
        return {
            "status": "skipped",
            "message": "ORNL scraping temporarily disabled",
            "data": []
        }
