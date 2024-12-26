import requests
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper
import os

class InspireScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.base_url = "https://inspirehep.net/api"
        self.search_terms = [
            'warp drive',
            'alcubierre metric',
            'quantum vacuum',
            'negative energy density',
            'exotic matter',
            'casimir effect',
            'spacetime metric engineering',
            'faster than light',
            'quantum tunneling',
            'quantum field theory propulsion',
            'quantum chromodynamics vacuum',
            'quantum electrodynamics vacuum',
            'zero point energy',
            'quantum vacuum fluctuations',
            'dark energy propulsion'
        ]
        
    def fetch_papers(self, search_term: str) -> List[Dict[str, Any]]:
        """Fetch papers related to a specific search term"""
        endpoint = f"{self.base_url}/literature"
        params = {
            'q': search_term,
            'size': 100,
            'sort': 'mostrecent',
            'fields': [
                'titles',
                'abstracts',
                'authors',
                'arxiv_eprints',
                'dois',
                'publication_info',
                'keywords',
                'inspire_categories',
                'earliest_date'
            ]
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            return response.json()['hits']['hits']
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching papers for term '{search_term}': {str(e)}")
            return []

    def fetch_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all available INSPIRE-HEP data"""
        self.logger.info("Fetching INSPIRE-HEP Data")
        
        all_papers = {}
        for term in self.search_terms:
            papers = self.fetch_papers(term)
            if papers:
                all_papers[term] = papers
                
        return all_papers

    def preprocess_data(self, data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess INSPIRE-HEP data"""
        self.logger.info("Preprocessing INSPIRE-HEP data")
        
        dfs = []
        
        for search_term, papers in data.items():
            for paper in papers:
                metadata = paper.get('metadata', {})
                
                # Extract title
                title = metadata.get('titles', [{}])[0].get('title', '')
                
                # Extract abstract
                abstract = ''
                if 'abstracts' in metadata and metadata['abstracts']:
                    abstract = metadata['abstracts'][0].get('value', '')
                
                # Extract authors
                authors = []
                if 'authors' in metadata:
                    authors = [author.get('full_name', '') for author in metadata['authors']]
                
                # Extract DOI
                doi = ''
                if 'dois' in metadata and metadata['dois']:
                    doi = metadata['dois'][0].get('value', '')
                
                # Extract arXiv ID
                arxiv_id = ''
                if 'arxiv_eprints' in metadata and metadata['arxiv_eprints']:
                    arxiv_id = metadata['arxiv_eprints'][0].get('value', '')
                
                # Extract keywords
                keywords = []
                if 'keywords' in metadata:
                    keywords = [kw.get('value', '') for kw in metadata['keywords']]
                
                # Extract categories
                categories = []
                if 'inspire_categories' in metadata:
                    categories = [cat.get('term', '') for cat in metadata['inspire_categories']]
                
                df = pd.DataFrame([{
                    'title': title,
                    'description': abstract,
                    'authors': ', '.join(authors),
                    'date': metadata.get('earliest_date', ''),
                    'doi': doi,
                    'arxiv_id': arxiv_id,
                    'keywords': ', '.join(keywords),
                    'categories': ', '.join(categories),
                    'search_term': search_term,
                    'data_type': 'research_paper'
                }])
                
                dfs.append(df)
        
        # Combine all datasets
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.DataFrame()
        
        # Add metadata
        df['source'] = 'INSPIRE-HEP'
        df['scrape_timestamp'] = pd.Timestamp.now()
        
        # Remove duplicates based on title
        df = df.drop_duplicates(subset=['title'])
        
        self.logger.info("Preprocessing completed")
        return df
