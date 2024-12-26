import requests
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper
from bs4 import BeautifulSoup
import time

class IEEEScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.base_url = "https://ieeexplore.ieee.org"
        self.search_terms = [
            'quantum propulsion',
            'magnetic containment field',
            'plasma containment',
            'energy field generation',
            'spacetime engineering',
            'quantum field manipulation',
            'advanced propulsion systems',
            'field generator design',
            'inertial dampening',
            'radiation shielding technology',
            'quantum vacuum engineering',
            'electromagnetic field control',
            'high energy materials',
            'metamaterial applications'
        ]
        
    def fetch_papers(self, search_term: str) -> List[Dict[str, Any]]:
        """Fetch papers from IEEE Xplore"""
        papers = []
        url = f"{self.base_url}/rest/search"
        
        try:
            # Add delay to be nice to the server
            time.sleep(2)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            
            params = {
                'queryText': search_term,
                'highlight': 'true',
                'returnFacets': 'ALL',
                'returnType': 'SEARCH',
                'sortType': 'newest'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for paper in data.get('records', []):
                papers.append({
                    'title': paper.get('title', ''),
                    'authors': paper.get('authors', []),
                    'abstract': paper.get('abstract', ''),
                    'publication': paper.get('publicationTitle', ''),
                    'date': paper.get('publicationDate', ''),
                    'doi': paper.get('doi', ''),
                    'url': f"{self.base_url}{paper.get('documentLink', '')}",
                    'keywords': paper.get('keywords', []),
                    'citations': paper.get('citationCount', 0)
                })
                
        except Exception as e:
            self.logger.error(f"Error fetching IEEE papers for term '{search_term}': {str(e)}")
            
        return papers

    def fetch_technical_standards(self, search_term: str) -> List[Dict[str, Any]]:
        """Fetch technical standards from IEEE"""
        standards = []
        url = f"{self.base_url}/rest/standards"
        
        try:
            time.sleep(2)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            
            params = {
                'queryText': search_term,
                'returnFacets': 'ALL'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for standard in data.get('standards', []):
                standards.append({
                    'title': standard.get('title', ''),
                    'standard_number': standard.get('standardNumber', ''),
                    'abstract': standard.get('abstract', ''),
                    'status': standard.get('status', ''),
                    'date': standard.get('publicationDate', ''),
                    'url': f"{self.base_url}{standard.get('documentLink', '')}",
                    'keywords': standard.get('keywords', [])
                })
                
        except Exception as e:
            self.logger.error(f"Error fetching IEEE standards for term '{search_term}': {str(e)}")
            
        return standards

    def fetch_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all available IEEE data"""
        self.logger.info("Fetching IEEE data")
        
        all_data = {}
        for term in self.search_terms:
            results = []
            
            # Fetch papers
            papers = self.fetch_papers(term)
            if papers:
                results.extend([{**paper, 'type': 'paper'} for paper in papers])
            
            # Fetch standards
            standards = self.fetch_technical_standards(term)
            if standards:
                results.extend([{**standard, 'type': 'standard'} for standard in standards])
            
            if results:
                all_data[term] = results
                
        return all_data

    def preprocess_data(self, data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess IEEE data"""
        self.logger.info("Preprocessing IEEE data")
        
        dfs = []
        
        for search_term, results in data.items():
            for result in results:
                # Convert authors list to string if needed
                authors = result.get('authors', [])
                if isinstance(authors, list):
                    authors = ', '.join(authors)
                
                # Convert keywords list to string if needed
                keywords = result.get('keywords', [])
                if isinstance(keywords, list):
                    keywords = ', '.join(keywords)
                
                df = pd.DataFrame([{
                    'title': result.get('title', ''),
                    'authors': authors,
                    'description': result.get('abstract', ''),
                    'publication': result.get('publication', ''),
                    'date': result.get('date', ''),
                    'url': result.get('url', ''),
                    'keywords': keywords,
                    'citations': result.get('citations', 0),
                    'type': result.get('type', ''),
                    'search_term': search_term,
                    'source': 'IEEE'
                }])
                
                dfs.append(df)
        
        # Combine all datasets
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.DataFrame()
        
        # Add metadata
        df['scrape_timestamp'] = pd.Timestamp.now()
        
        # Remove duplicates based on URL
        df = df.drop_duplicates(subset=['url'])
        
        self.logger.info("Preprocessing completed")
        return df
