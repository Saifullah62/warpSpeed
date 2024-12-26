import requests
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper
import json

class EHTScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.base_url = "https://eventhorizontelescope.org/data"
        self.api_endpoint = "https://eventhorizontelescope.org/api/public/data"
        
    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch black hole imaging data from Event Horizon Telescope"""
        self.logger.info("Fetching EHT black hole imaging data")
        
        try:
            response = requests.get(self.api_endpoint)
            response.raise_for_status()
            
            data = response.json()
            observations = []
            
            for item in data:
                observation = {
                    'target_name': item.get('target'),
                    'observation_date': item.get('date'),
                    'wavelength': item.get('wavelength'),
                    'resolution': item.get('angular_resolution'),
                    'data_url': item.get('download_url'),
                    'observation_type': item.get('type')
                }
                observations.append(observation)
            
            self.logger.info(f"Successfully fetched {len(observations)} observations")
            return observations
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching EHT data: {str(e)}")
            raise
            
    def preprocess_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Preprocess EHT black hole imaging data"""
        self.logger.info("Preprocessing EHT data")
        
        df = pd.DataFrame(data)
        
        # Convert observation date to datetime
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        
        # Extract wavelength value and unit
        df[['wavelength_value', 'wavelength_unit']] = df['wavelength'].str.extract(r'(\d+\.?\d*)\s*(\w+)')
        df['wavelength_value'] = pd.to_numeric(df['wavelength_value'])
        
        # Add metadata
        df['source'] = 'EHT'
        df['scrape_timestamp'] = pd.Timestamp.now()
        
        self.logger.info("Preprocessing completed")
        return df
