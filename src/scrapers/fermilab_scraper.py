import requests
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper
import os

class FermilabScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.base_url = "https://opendata.fnal.gov/api/v1"
        self.api_key = os.getenv('FERMILAB_API_KEY')
        
    def fetch_quantum_experiments(self) -> List[Dict[str, Any]]:
        """Fetch quantum physics experiment data"""
        endpoint = f"{self.base_url}/experiments"
        params = {
            'type': 'quantum',
            'api_key': self.api_key,
            'limit': 100
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()['experiments']
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching quantum experiments: {str(e)}")
            raise

    def fetch_particle_data(self) -> List[Dict[str, Any]]:
        """Fetch particle physics data"""
        endpoint = f"{self.base_url}/particles"
        params = {
            'api_key': self.api_key,
            'limit': 100
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()['particles']
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching particle data: {str(e)}")
            raise

    def fetch_accelerator_data(self) -> List[Dict[str, Any]]:
        """Fetch accelerator experiment data"""
        endpoint = f"{self.base_url}/accelerator"
        params = {
            'api_key': self.api_key,
            'limit': 100
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()['data']
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching accelerator data: {str(e)}")
            raise

    def fetch_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all available Fermilab data"""
        self.logger.info("Fetching Fermilab data")
        
        data = {
            'quantum_experiments': self.fetch_quantum_experiments(),
            'particle_data': self.fetch_particle_data(),
            'accelerator_data': self.fetch_accelerator_data()
        }
        
        return data
            
    def preprocess_data(self, data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess Fermilab data"""
        self.logger.info("Preprocessing Fermilab data")
        
        # Process quantum experiments
        quantum_df = pd.DataFrame(data['quantum_experiments'])
        if not quantum_df.empty:
            quantum_df['data_type'] = 'quantum_experiment'
            
        # Process particle data
        particle_df = pd.DataFrame(data['particle_data'])
        if not particle_df.empty:
            particle_df['data_type'] = 'particle_data'
            
        # Process accelerator data
        accelerator_df = pd.DataFrame(data['accelerator_data'])
        if not accelerator_df.empty:
            accelerator_df['data_type'] = 'accelerator_data'
        
        # Combine all datasets
        df = pd.concat(
            [quantum_df, particle_df, accelerator_df],
            ignore_index=True
        )
        
        # Convert dates if present
        date_columns = ['date', 'start_date', 'end_date']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Add metadata
        df['source'] = 'Fermilab'
        df['scrape_timestamp'] = pd.Timestamp.now()
        
        # Flag exotic particle experiments
        keywords = ['exotic', 'dark matter', 'antimatter', 'strange quark',
                   'tachyon', 'monopole', 'supersymmetry']
                   
        def check_exotic(text):
            if pd.isna(text):
                return False
            return any(kw in text.lower() for kw in keywords)
        
        if 'description' in df.columns:
            df['is_exotic'] = df['description'].apply(check_exotic)
        
        self.logger.info("Preprocessing completed")
        return df
