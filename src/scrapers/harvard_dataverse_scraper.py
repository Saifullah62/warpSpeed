import requests
import pandas as pd
from typing import Dict, Any, List
from src.scrapers.base_scraper import BaseScraper
from bs4 import BeautifulSoup
import time
import re
from src.utils.retry_utils import (
    with_retry, handle_http_error, safe_get_text,
    safe_get_attr, validate_response
)
from requests.exceptions import RequestException

class HarvardDataverseScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.base_url = "https://dataverse.harvard.edu"
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

    def _extract_dataset_data(self, element: BeautifulSoup) -> Dict[str, Any]:
        """Safely extract dataset data from HTML element."""
        try:
            return {
                'title': safe_get_text(element.find('h4', class_='card-title')),
                'authors': [
                    author.text.strip() 
                    for author in element.find_all('a', class_='author-link')
                ],
                'description': safe_get_text(element.find('div', class_='description')),
                'publication_date': safe_get_text(element.find('span', class_='date')),
                'url': self.base_url + safe_get_attr(element.find('a', class_='card-title-link'), 'href')
            }
        except Exception as e:
            self.logger.error(f"Error extracting dataset data: {str(e)}")
            return {}

    def fetch_black_hole_datasets(self) -> List[Dict[str, Any]]:
        """Fetch black hole research datasets using Harvard Dataverse API"""
        search_url = f"{self.base_url}/api/search"
        params = {
            'q': 'black hole physics OR astrophysics',
            'type': 'dataset',
            'sort': 'date',
            'order': 'desc',
            'per_page': 100,
            'show_relevance': True,
            'show_facets': True
        }
        datasets = []
        
        try:
            response = self._make_request(f"{search_url}?{requests.compat.urlencode(params)}")
            data = response.json()
            
            # Extract items from the response
            items = data.get('data', {}).get('items', [])
            if not items:
                self.logger.warning("No items found in API response")
                return []
                
            for item in items:
                try:
                    # Skip if item is not a dict
                    if not isinstance(item, dict):
                        continue
                        
                    # Extract authors safely
                    authors = []
                    for author in item.get('authors', []):
                        if isinstance(author, dict):
                            author_name = author.get('name', '')
                            if author_name:
                                authors.append(author_name)
                    
                    dataset = {
                        'title': item.get('name', ''),
                        'authors': authors,
                        'description': item.get('description', ''),
                        'publication_date': item.get('published_at', ''),
                        'url': f"{self.base_url}/dataset.xhtml?persistentId={item.get('global_id', '')}",
                        'subject': item.get('subject', ''),
                        'keywords': item.get('keywords', []),
                        'file_count': item.get('file_count', 0),
                        'size_bytes': item.get('size', 0),
                        'version': item.get('version', ''),
                        'license': item.get('license', ''),
                        'data_type': item.get('data_type', ''),
                        'citation': item.get('citation', '')
                    }
                    
                    # Basic validation
                    if dataset['title'] and dataset['description']:
                        datasets.append(dataset)
                        
                except Exception as e:
                    self.logger.error(f"Error extracting dataset data: {str(e)}")
                    continue
                    
            if not datasets:
                self.logger.warning("No valid datasets found after filtering")
                
            return datasets
            
        except Exception as e:
            self.logger.error(f"Error in fetch_black_hole_datasets: {str(e)}")
            return []

    def fetch_theoretical_models(self) -> List[Dict[str, Any]]:
        """Fetch theoretical physics models using Harvard Dataverse API"""
        search_url = f"{self.base_url}/api/search"
        params = {
            'q': 'theoretical physics model OR simulation',
            'type': 'dataset',
            'sort': 'date',
            'order': 'desc',
            'per_page': 100,
            'show_relevance': True,
            'show_facets': True
        }
        models = []
        
        try:
            response = self._make_request(f"{search_url}?{requests.compat.urlencode(params)}")
            data = response.json()
            
            # Extract items from the response
            items = data.get('data', {}).get('items', [])
            if not items:
                self.logger.warning("No items found in API response")
                return []
                
            for item in items:
                try:
                    # Skip if item is not a dict
                    if not isinstance(item, dict):
                        continue
                        
                    # Only include theoretical physics models
                    description = item.get('description', '').lower()
                    if any(kw in description for kw in ['theoretical', 'model', 'simulation']):
                        # Extract authors safely
                        authors = []
                        for author in item.get('authors', []):
                            if isinstance(author, dict):
                                author_name = author.get('name', '')
                                if author_name:
                                    authors.append(author_name)
                        
                        model = {
                            'title': item.get('name', ''),
                            'authors': authors,
                            'description': item.get('description', ''),
                            'publication_date': item.get('published_at', ''),
                            'url': f"{self.base_url}/dataset.xhtml?persistentId={item.get('global_id', '')}",
                            'subject': item.get('subject', ''),
                            'keywords': item.get('keywords', []),
                            'file_count': item.get('file_count', 0),
                            'size_bytes': item.get('size', 0),
                            'version': item.get('version', '')
                        }
                        
                        # Basic validation
                        if model['title'] and model['description']:
                            models.append(model)
                            
                except Exception as e:
                    self.logger.error(f"Error extracting model data: {str(e)}")
                    continue
                    
            if not models:
                self.logger.warning("No valid models found after filtering")
                
            return models
            
        except Exception as e:
            self.logger.error(f"Error in fetch_theoretical_models: {str(e)}")
            return []

    @with_retry(max_attempts=3, base_delay=2.0)
    def fetch_dataset_details(self, url: str) -> Dict[str, Any]:
        """Fetch detailed information for a specific dataset using API"""
        try:
            # Extract persistent ID from URL
            persistent_id = url.split('persistentId=')[-1]
            api_url = f"{self.base_url}/api/datasets/:persistentId/?persistentId={persistent_id}"
            
            response = self._make_request(api_url)
            data = response.json()
            dataset = data.get('data', {})
            
            # Extract metadata
            metadata = dataset.get('latestVersion', {}).get('metadataBlocks', {}).get('citation', {}).get('fields', [])
            
            details = {
                'title': '',
                'authors': [],
                'description': '',
                'publication_date': '',
                'keywords': [],
                'methodology': '',
                'data_source': '',
                'license': '',
                'citation': '',
                'related_publications': []
            }
            
            # Parse metadata fields
            for field in metadata:
                field_name = field.get('typeName', '')
                if field_name == 'title':
                    details['title'] = field.get('value', '')
                elif field_name == 'author':
                    details['authors'] = [
                        author.get('authorName', {}).get('value', '')
                        for author in field.get('value', [])
                    ]
                elif field_name == 'dsDescription':
                    details['description'] = ' '.join(
                        desc.get('dsDescriptionValue', {}).get('value', '')
                        for desc in field.get('value', [])
                    )
                elif field_name == 'keyword':
                    details['keywords'] = [
                        kw.get('keywordValue', {}).get('value', '')
                        for kw in field.get('value', [])
                    ]
                elif field_name == 'dataSources':
                    details['data_source'] = ' '.join(
                        source.get('dataSource', {}).get('value', '')
                        for source in field.get('value', [])
                    )
                elif field_name == 'publicationDate':
                    details['publication_date'] = field.get('value', '')
                elif field_name == 'license':
                    details['license'] = field.get('value', '')
            
            # Get files information
            files = dataset.get('latestVersion', {}).get('files', [])
            details['files'] = [
                {
                    'name': file.get('dataFile', {}).get('filename', ''),
                    'size': file.get('dataFile', {}).get('filesize', 0),
                    'type': file.get('dataFile', {}).get('contentType', ''),
                    'description': file.get('dataFile', {}).get('description', '')
                }
                for file in files
            ]
            
            return details
            
        except Exception as e:
            self.logger.error(f"Error fetching dataset details for {url}: {str(e)}")
            return {}

    def fetch_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all relevant Harvard Dataverse data"""
        self.logger.info("Fetching Harvard Dataverse data")
        
        data = {
            'black_hole_research': self.fetch_black_hole_datasets(),
            'theoretical_models': self.fetch_theoretical_models()
        }
        
        # Enrich with detailed information
        for category in data:
            for i, item in enumerate(data[category]):
                if 'url' in item and item['url']:
                    details = self.fetch_dataset_details(item['url'])
                    data[category][i].update(details)
                    time.sleep(1)
        
        return data
            
    def preprocess_data(self, data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess Harvard Dataverse data"""
        self.logger.info("Preprocessing Harvard Dataverse data")
        
        try:
            # Process black hole research data
            black_hole_df = pd.DataFrame(data['black_hole_research'])
            if not black_hole_df.empty:
                black_hole_df['research_category'] = 'black_hole'
                
            # Process theoretical models
            theory_df = pd.DataFrame(data['theoretical_models'])
            if not theory_df.empty:
                theory_df['research_category'] = 'theoretical_model'
            
            # Combine datasets
            df = pd.concat([black_hole_df, theory_df], ignore_index=True)
            
            # Convert dates safely
            if 'publication_date' in df.columns:
                df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
            
            # Add metadata
            df['source'] = 'Harvard_Dataverse'
            df['scrape_timestamp'] = pd.Timestamp.now()
            
            # Categorize research focus
            keywords_categories = {
                'event_horizon': ['event horizon', 'horizon detection', 'horizon physics'],
                'singularity': ['singularity', 'spacetime singularity'],
                'quantum_effects': ['quantum gravity', 'quantum effects', 'hawking radiation'],
                'wormhole': ['wormhole', 'einstein-rosen bridge'],
                'theoretical': ['theoretical model', 'simulation', 'numerical relativity']
            }
            
            def categorize_research(text):
                if pd.isna(text):
                    return []
                text = str(text).lower()
                categories = []
                for category, keywords in keywords_categories.items():
                    if any(kw in text for kw in keywords):
                        categories.append(category)
                return categories
            
            if 'description' in df.columns:
                df['research_focus'] = df['description'].apply(categorize_research)
            
            # Calculate impact metrics safely
            if all(col in df.columns for col in ['citations', 'downloads']):
                df['impact_score'] = (
                    pd.to_numeric(df['citations'], errors='coerce').fillna(0) * 2 + 
                    pd.to_numeric(df['downloads'], errors='coerce').fillna(0) * 0.1
                )
            
            self.logger.info("Preprocessing completed")
            return df
            
        except Exception as e:
            self.logger.error(f"Error in preprocess_data: {str(e)}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(
                columns=['title', 'authors', 'description', 'research_category', 
                        'source', 'scrape_timestamp']
            )
