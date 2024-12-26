import requests
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper
from astropy.coordinates import SkyCoord
import astropy.units as u

class SIMBADScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.base_url = "http://simbad.u-strasbg.fr/simbad/sim-tap/sync"
        
    def fetch_exotic_objects(self) -> List[Dict[str, Any]]:
        """Fetch data about exotic astronomical objects"""
        query = """
        SELECT 
            basic.OID, 
            basic.MAIN_ID, 
            otypes.OTYPE_NAME, 
            basic.RA, 
            basic.DEC,
            mesDistance.Distance_value,
            mesDistance.Distance_unit
        FROM 
            basic 
            LEFT JOIN otypes ON basic.OTYPE = otypes.OTYPE
            LEFT JOIN mesDistance ON basic.OID = mesDistance.OID
        WHERE 
            otypes.OTYPE_NAME IN (
                'Black_Hole', 
                'Neutron_Star',
                'Quasar',
                'Gravitational_Lens',
                'Dark_Matter_Halo'
            )
        LIMIT 1000
        """
        
        params = {
            'query': query,
            'format': 'json'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()['data']
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching exotic objects: {str(e)}")
            raise

    def fetch_high_energy_objects(self) -> List[Dict[str, Any]]:
        """Fetch data about high-energy astronomical objects"""
        query = """
        SELECT 
            basic.OID,
            basic.MAIN_ID,
            otypes.OTYPE_NAME,
            basic.RA,
            basic.DEC,
            mesFlux.Flux_value,
            mesFlux.Flux_unit,
            mesFlux.Filter_name
        FROM 
            basic
            LEFT JOIN otypes ON basic.OTYPE = otypes.OTYPE
            LEFT JOIN mesFlux ON basic.OID = mesFlux.OID
        WHERE 
            mesFlux.Filter_name IN ('X', 'GAMMA')
            AND mesFlux.Flux_value > 0
        LIMIT 1000
        """
        
        params = {
            'query': query,
            'format': 'json'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()['data']
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching high-energy objects: {str(e)}")
            raise

    def fetch_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all relevant SIMBAD data"""
        self.logger.info("Fetching SIMBAD astronomical data")
        
        data = {
            'exotic_objects': self.fetch_exotic_objects(),
            'high_energy_objects': self.fetch_high_energy_objects()
        }
        
        return data
            
    def preprocess_data(self, data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess SIMBAD astronomical data"""
        self.logger.info("Preprocessing SIMBAD data")
        
        # Process exotic objects
        exotic_df = pd.DataFrame(data['exotic_objects'])
        if not exotic_df.empty:
            exotic_df['data_type'] = 'exotic_object'
            
        # Process high-energy objects
        high_energy_df = pd.DataFrame(data['high_energy_objects'])
        if not high_energy_df.empty:
            high_energy_df['data_type'] = 'high_energy_object'
        
        # Combine datasets
        df = pd.concat([exotic_df, high_energy_df], ignore_index=True)
        
        # Convert coordinates to SkyCoord objects
        if 'RA' in df.columns and 'DEC' in df.columns:
            coords = SkyCoord(
                ra=df['RA'].values * u.degree,
                dec=df['DEC'].values * u.degree
            )
            df['galactic_l'] = coords.galactic.l.degree
            df['galactic_b'] = coords.galactic.b.degree
        
        # Convert distance to parsecs if available
        if 'Distance_value' in df.columns and 'Distance_unit' in df.columns:
            def convert_to_parsec(row):
                if pd.isna(row['Distance_value']) or pd.isna(row['Distance_unit']):
                    return None
                value = float(row['Distance_value'])
                unit = row['Distance_unit'].lower()
                if unit == 'pc':
                    return value
                elif unit == 'kpc':
                    return value * 1000
                elif unit == 'mpc':
                    return value * 1000000
                return None
            
            df['distance_parsec'] = df.apply(convert_to_parsec, axis=1)
        
        # Add metadata
        df['source'] = 'SIMBAD'
        df['scrape_timestamp'] = pd.Timestamp.now()
        
        # Calculate additional metrics
        if 'distance_parsec' in df.columns:
            df['distance_category'] = pd.qcut(
                df['distance_parsec'].fillna(df['distance_parsec'].mean()),
                q=4,
                labels=['nearby', 'intermediate', 'distant', 'very_distant']
            )
        
        self.logger.info("Preprocessing completed")
        return df
