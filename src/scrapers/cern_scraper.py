import requests
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper
import os

class CERNScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.api_key = os.getenv('CERN_API_KEY')
        self.base_url = "http://opendata.cern.ch/api"
        self.search_terms = [
            'quantum field',
            'vacuum state',
            'quantum chromodynamics',
            'quantum electrodynamics',
            'higgs field',
            'dark energy',
            'antimatter',
            'particle physics propulsion',
            'quantum vacuum fluctuations',
            'zero point energy',
            'casimir effect',
            'quantum tunneling',
            'quantum entanglement'
        ]
        
    def fetch_theoretical_papers(self) -> List[Dict[str, Any]]:
        """Fetch theoretical physics papers"""
        endpoint = f"{self.base_url}/records"
        params = {
            'type': 'Publication',
            'subtype': 'Theory',
            'q': ' OR '.join(self.search_terms),
            'size': 100
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            return response.json()['hits']['hits']
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching theoretical papers: {str(e)}")
            return []

    def fetch_experiments(self) -> List[Dict[str, Any]]:
        """Fetch data from CERN experiments"""
        endpoint = f"{self.base_url}/records"
        params = {
            'type': 'Dataset',
            'experiment': ['CMS', 'ATLAS', 'ALICE', 'LHCb'],
            'q': ' OR '.join(self.search_terms),
            'size': 100
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            return response.json()['hits']['hits']
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching CERN experiment data: {str(e)}")
            return []

    def fetch_quantum_measurements(self) -> List[Dict[str, Any]]:
        """Fetch quantum-related measurements"""
        endpoint = f"{self.base_url}/records"
        params = {
            'type': 'Dataset',
            'subtype': 'Derived',
            'q': 'quantum OR vacuum OR field theory',
            'size': 100
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            return response.json()['hits']['hits']
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching quantum measurements: {str(e)}")
            return []

    def fetch_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all available CERN data"""
        self.logger.info("Fetching CERN Open Data")
        
        return {
            'theoretical_papers': self.fetch_theoretical_papers(),
            'experiments': self.fetch_experiments(),
            'quantum_measurements': self.fetch_quantum_measurements()
        }

    def preprocess_data(self, data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess CERN data"""
        self.logger.info("Preprocessing CERN data")
        
        dfs = []
        
        # Process all datasets
        for data_type, records in data.items():
            if records:
                for record in records:
                    metadata = record.get('metadata', {})
                    df = pd.DataFrame([{
                        'title': metadata.get('title', ''),
                        'description': metadata.get('description', ''),
                        'date': metadata.get('date_published', ''),
                        'authors': ', '.join(metadata.get('authors', [])),
                        'keywords': ', '.join(metadata.get('keywords', [])),
                        'doi': metadata.get('doi', ''),
                        'data_type': data_type,
                        'experiment': metadata.get('experiment', ''),
                        'url': f"http://opendata.cern.ch/record/{record.get('id', '')}"
                    }])
                    dfs.append(df)
        
        # Combine all datasets
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.DataFrame()
        
        # Add metadata
        df['source'] = 'CERN'
        df['scrape_timestamp'] = pd.Timestamp.now()
        
        self.logger.info("Preprocessing completed")
        return df
