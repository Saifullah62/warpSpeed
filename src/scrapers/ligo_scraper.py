import requests
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper

class LIGOScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.base_url = "https://www.gw-openscience.org/catalog/GWTC-1-confident/html/"
        
    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch gravitational wave event data from LIGO"""
        self.logger.info("Fetching LIGO gravitational wave events")
        
        try:
            response = requests.get(self.base_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            events = []
            
            # Find the table containing gravitational wave events
            table = soup.find('table', {'class': 'events'})
            if table:
                rows = table.find_all('tr')[1:]  # Skip header row
                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 6:
                        event = {
                            'event_name': cols[0].text.strip(),
                            'gps_time': cols[1].text.strip(),
                            'mass_1_solar': cols[2].text.strip(),
                            'mass_2_solar': cols[3].text.strip(),
                            'network_snr': cols[4].text.strip(),
                            'false_alarm_rate': cols[5].text.strip()
                        }
                        events.append(event)
            
            self.logger.info(f"Successfully fetched {len(events)} events")
            return events
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching LIGO data: {str(e)}")
            raise
            
    def preprocess_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Preprocess LIGO gravitational wave data"""
        self.logger.info("Preprocessing LIGO data")
        
        df = pd.DataFrame(data)
        
        # Convert numeric columns
        numeric_cols = ['mass_1_solar', 'mass_2_solar', 'network_snr']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Convert GPS time to datetime
        df['datetime'] = pd.to_datetime(df['gps_time'], unit='s')
        
        # Add metadata
        df['source'] = 'LIGO'
        df['scrape_timestamp'] = pd.Timestamp.now()
        
        self.logger.info("Preprocessing completed")
        return df
