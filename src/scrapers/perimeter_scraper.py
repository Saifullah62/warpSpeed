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

class PerimeterScraper(BaseScraper):
    def __init__(self, output_dir: str):
        """Initialize the Perimeter Institute scraper"""
        super().__init__(output_dir)
        self.base_url = "https://perimeterinstitute.ca"
        self.backup_urls = {
            'research': [
                '/research/research-areas',
                '/research/quantum-matter',
                '/research/quantum-fields-strings',
                '/research/quantum-information',
                '/research/cosmology'
            ],
            'seminars': [
                '/outreach/public-lectures',
                '/training/seminars',
                '/research/conferences-and-workshops',
                '/video-library'
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

    def fetch_quantum_gravity_research(self) -> List[Dict[str, Any]]:
        """Fetch quantum gravity research papers from Perimeter Institute website"""
        papers = []
        
        try:
            response = self._try_urls(self.backup_urls['research'])
            if not response:
                self.logger.error("Failed to fetch research data from any URL")
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try multiple possible container classes
            articles = (
                soup.find_all('div', class_='research-publication') or
                soup.find_all('div', class_='publication') or
                soup.find_all('article') or
                soup.find_all('div', class_='content-item')
            )
            
            for article in articles:
                # Look for quantum gravity related content
                title = safe_get_text(article, ['h2', 'h3', 'h4']).lower()
                if any(kw in title for kw in ['quantum', 'gravity', 'spacetime']):
                    data = {
                        'title': safe_get_text(article, ['h2', 'h3', 'h4']),
                        'abstract': safe_get_text(article, ['div.abstract', 'div.description', 'p']),
                        'authors': [
                            author.text.strip() 
                            for author in article.find_all(['span', 'div'], class_=['author', 'researcher'])
                        ],
                        'date': safe_get_text(article, ['div.date', 'time', 'span.date']),
                        'url': self.base_url + safe_get_attr(article, 'a', 'href'),
                        'arxiv_id': safe_get_text(article, ['span.arxiv', 'div.arxiv-id']),
                        'journal': safe_get_text(article, ['span.journal', 'div.journal-ref'])
                    }
                    if data['title']:
                        papers.append(data)
            
            return papers
            
        except Exception as e:
            self.logger.error(f"Error fetching quantum gravity research: {str(e)}")
            return []

    def fetch_research_areas(self) -> List[Dict[str, Any]]:
        """Fetch information about research areas from Perimeter Institute website"""
        research_areas = []
        
        try:
            response = self._try_urls(self.backup_urls['research'])
            if not response:
                self.logger.error("Failed to fetch research areas from any URL")
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try multiple possible container classes
            areas = (
                soup.find_all('div', class_='research-area') or
                soup.find_all('div', class_='field') or
                soup.find_all('article', class_='area') or
                soup.find_all('div', class_='content-item')
            )
            
            for area in areas:
                data = {
                    'name': safe_get_text(area, ['h2', 'h3', 'h4']),
                    'description': safe_get_text(area, ['div.description', 'div.summary', 'p']),
                    'researchers': [
                        researcher.text.strip() 
                        for researcher in area.find_all(['span', 'div'], class_=['researcher', 'faculty'])
                    ],
                    'url': self.base_url + safe_get_attr(area, 'a', 'href')
                }
                if data['name']:
                    research_areas.append(data)
            
            return research_areas
            
        except Exception as e:
            self.logger.error(f"Error fetching research areas: {str(e)}")
            return []

    def fetch_seminar_data(self) -> List[Dict[str, Any]]:
        """Fetch data from research seminars using Perimeter Institute website"""
        seminars = []
        
        try:
            response = self._try_urls(self.backup_urls['seminars'])
            if not response:
                self.logger.error("Failed to fetch seminar data from any URL")
                return []

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Try multiple possible container classes
            seminar_containers = (
                soup.find_all('div', class_='seminar') or
                soup.find_all('div', class_='event') or
                soup.find_all('article', class_='talk') or
                soup.find_all('div', class_='content-item')
            )
            
            for seminar_container in seminar_containers:
                # Look for physics related content
                title = safe_get_text(seminar_container, ['h2', 'h3', 'h4']).lower()
                if any(kw in title for kw in ['quantum', 'gravity', 'physics', 'theory']):
                    data = {
                        'title': safe_get_text(seminar_container, ['h2', 'h3', 'h4']),
                        'speaker': safe_get_text(seminar_container, ['div.speaker', 'span.speaker']),
                        'abstract': safe_get_text(seminar_container, ['div.abstract', 'div.description', 'p']),
                        'date': safe_get_text(seminar_container, ['div.date', 'time', 'span.date']),
                        'url': self.base_url + safe_get_attr(seminar_container, 'a', 'href'),
                        'video_url': safe_get_attr(seminar_container, ['a.video', 'a.recording'], 'href'),
                        'slides_url': safe_get_attr(seminar_container, ['a.slides', 'a.presentation'], 'href'),
                        'institution': safe_get_text(seminar_container, ['div.institution', 'span.affiliation'])
                    }
                    if data['title']:
                        seminars.append(data)
            
            return seminars
            
        except Exception as e:
            self.logger.error(f"Error fetching seminar data: {str(e)}")
            return []

    def fetch_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all relevant Perimeter Institute data"""
        self.logger.info("Fetching Perimeter Institute data")
        
        data = {
            'quantum_gravity': self.fetch_quantum_gravity_research(),
            'research_areas': self.fetch_research_areas(),
            'seminars': self.fetch_seminar_data()
        }
        
        return data
            
    def preprocess_data(self, data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess Perimeter Institute data"""
        self.logger.info("Preprocessing Perimeter Institute data")
        
        try:
            # Process quantum gravity research
            gravity_df = pd.DataFrame(data['quantum_gravity'])
            if not gravity_df.empty:
                gravity_df['content_type'] = 'research'
                
            # Process research areas
            areas_df = pd.DataFrame(data['research_areas'])
            if not areas_df.empty:
                areas_df['content_type'] = 'research_area'
                
            # Process seminar data
            seminars_df = pd.DataFrame(data['seminars'])
            if not seminars_df.empty:
                seminars_df['content_type'] = 'seminar'
            
            # Combine datasets
            df = pd.concat(
                [gravity_df, areas_df, seminars_df],
                ignore_index=True
            )
            
            # Convert dates safely
            date_columns = ['date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Add metadata
            df['source'] = 'Perimeter_Institute'
            df['scrape_timestamp'] = pd.Timestamp.now()
            
            self.logger.info("Preprocessing completed")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in preprocess_data: {str(e)}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(
                columns=['title', 'authors', 'abstract', 'content_type', 
                        'source', 'scrape_timestamp']
            )
