import requests
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper
from bs4 import BeautifulSoup

class ChandraScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.base_url = "https://cda.harvard.edu/chaser/mainEntry.do"
        self.api_url = "https://cda.harvard.edu/api/v1"
        
    def fetch_black_hole_data(self) -> List[Dict[str, Any]]:
        """Fetch black hole observations from Chandra"""
        endpoint = f"{self.api_url}/observations"
        params = {
            'category': 'black_hole',
            'format': 'json',
            'maxresults': 100
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()['observations']
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching black hole data: {str(e)}")
            raise

    def fetch_neutron_star_data(self) -> List[Dict[str, Any]]:
        """Fetch neutron star observations"""
        endpoint = f"{self.api_url}/observations"
        params = {
            'category': 'neutron_star',
            'format': 'json',
            'maxresults': 100
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()['observations']
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching neutron star data: {str(e)}")
            raise

    def fetch_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all available Chandra data"""
        self.logger.info("Fetching Chandra X-Ray Observatory Data")
        
        data = {
            'black_holes': self.fetch_black_hole_data(),
            'neutron_stars': self.fetch_neutron_star_data()
        }
        
        return data
            
    def preprocess_data(self, data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess Chandra X-Ray data"""
        self.logger.info("Preprocessing Chandra data")
        
        # Process black hole data
        black_holes_df = pd.DataFrame([
            {
                'target_name': obs.get('target_name'),
                'observation_id': obs.get('obsid'),
                'observation_date': obs.get('date_obs'),
                'exposure_time': obs.get('exposure'),
                'ra': obs.get('ra'),
                'dec': obs.get('dec'),
                'object_type': 'black_hole',
                'instrument': obs.get('instrument'),
                'energy_band': obs.get('energy_range'),
                'data_url': obs.get('data_url')
            }
            for obs in data['black_holes']
        ])
        
        # Process neutron star data
        neutron_stars_df = pd.DataFrame([
            {
                'target_name': obs.get('target_name'),
                'observation_id': obs.get('obsid'),
                'observation_date': obs.get('date_obs'),
                'exposure_time': obs.get('exposure'),
                'ra': obs.get('ra'),
                'dec': obs.get('dec'),
                'object_type': 'neutron_star',
                'instrument': obs.get('instrument'),
                'energy_band': obs.get('energy_range'),
                'data_url': obs.get('data_url')
            }
            for obs in data['neutron_stars']
        ])
        
        # Combine datasets
        df = pd.concat([black_holes_df, neutron_stars_df], ignore_index=True)
        
        # Convert dates to datetime
        df['observation_date'] = pd.to_datetime(df['observation_date'])
        
        # Convert numeric fields
        numeric_cols = ['exposure_time', 'ra', 'dec']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Parse energy band into min and max values
        def parse_energy_range(energy_range):
            if pd.isna(energy_range):
                return pd.NA, pd.NA
            try:
                min_e, max_e = map(float, energy_range.split('-'))
                return min_e, max_e
            except:
                return pd.NA, pd.NA
        
        df[['energy_min_kev', 'energy_max_kev']] = df['energy_band'].apply(
            lambda x: pd.Series(parse_energy_range(x))
        )
        
        # Add metadata
        df['source'] = 'Chandra'
        df['scrape_timestamp'] = pd.Timestamp.now()
        
        # Calculate additional metrics
        df['observation_duration_hours'] = df['exposure_time'] / 3600
        
        self.logger.info("Preprocessing completed")
        return df
