import requests
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper
import os

class NISTScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        # Hardcoded API key for private implementation
        self.api_key = "496fa992-9581-4b16-9740-5f21d9da5440"
        self.base_url = "https://data.nist.gov/rmm/records"
        
    def fetch_quantum_data(self) -> List[Dict[str, Any]]:
        """Fetch quantum science data from NIST"""
        params = {
            'searchphrase': '(quantum OR entanglement OR superposition) AND (propulsion OR drive OR energy)',
            'size': 100
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching NIST quantum data: {str(e)}")
            return []

    def fetch_vacuum_state_data(self) -> List[Dict[str, Any]]:
        """Fetch vacuum state and quantum fluctuation data"""
        params = {
            'searchphrase': '(vacuum state OR quantum fluctuation OR zero-point energy) AND (propulsion OR drive)',
            'size': 100
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching vacuum state data: {str(e)}")
            return []

    def fetch_materials_data(self) -> List[Dict[str, Any]]:
        """Fetch advanced materials data"""
        params = {
            'searchphrase': '(metamaterial OR exotic matter OR negative mass) AND (propulsion OR energy OR quantum)',
            'size': 100
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching materials data: {str(e)}")
            return []

    def fetch_measurement_data(self) -> List[Dict[str, Any]]:
        """Fetch precision measurement and standards data"""
        params = {
            'searchphrase': '(precision measurement OR quantum metrology OR atomic clock) AND (time OR space OR gravity)',
            'size': 100
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching measurement data: {str(e)}")
            return []

    def fetch_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all available NIST data"""
        self.logger.info("Fetching NIST Quantum Data")
        
        return {
            'quantum': self.fetch_quantum_data(),
            'vacuum': self.fetch_vacuum_state_data(),
            'materials': self.fetch_materials_data(),
            'measurements': self.fetch_measurement_data()
        }

    def preprocess_data(self, data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess NIST data"""
        self.logger.info("Preprocessing NIST data")
        
        dfs = []
        
        # Process all datasets
        for data_type, records in data.items():
            if records:
                df = pd.DataFrame(records)
                df['data_type'] = data_type
                df['title'] = df.get('title', '')
                df['description'] = df.apply(
                    lambda x: f"{x.get('description', '')} Keywords: {', '.join(x.get('keyword', []))}", 
                    axis=1
                )
                df['date'] = df.get('modified', '')
                dfs.append(df)
        
        # Combine all datasets
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.DataFrame()
        
        # Add metadata
        df['source'] = 'NIST'
        df['scrape_timestamp'] = pd.Timestamp.now()
        
        # Select and rename columns
        df = df[['title', 'description', 'date', 'data_type', 'source', 'scrape_timestamp']]
        
        self.logger.info("Preprocessing completed")
        return df
