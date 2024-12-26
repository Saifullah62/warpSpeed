import requests
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper
import h5py
import numpy as np
import tempfile
import os

class EinsteinToolkitScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.base_url = "https://einsteintoolkit.org/data"
        self.simulations_url = f"{self.base_url}/simulations"
        
    def fetch_simulation_metadata(self) -> List[Dict[str, Any]]:
        """Fetch metadata about available simulations"""
        try:
            response = requests.get(f"{self.simulations_url}/catalog.json")
            response.raise_for_status()
            return response.json()['simulations']
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching simulation metadata: {str(e)}")
            raise

    def download_simulation_data(self, simulation_id: str) -> Dict[str, Any]:
        """Download specific simulation data"""
        try:
            # Download simulation HDF5 file
            response = requests.get(f"{self.simulations_url}/{simulation_id}/data.h5")
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name
            
            # Extract data from HDF5 file
            with h5py.File(tmp_path, 'r') as f:
                data = {
                    'simulation_id': simulation_id,
                    'mass': f['parameters/mass'][()],
                    'spin': f['parameters/spin'][()],
                    'time_steps': len(f['evolution/time']),
                    'max_density': np.max(f['evolution/density']),
                    'min_lapse': np.min(f['evolution/lapse']),
                    'horizon_mass': f['horizons/mass'][-1] if 'horizons/mass' in f else None,
                    'gravitational_wave_strain': f['waves/strain'][-1] if 'waves/strain' in f else None
                }
            
            # Clean up temporary file
            os.unlink(tmp_path)
            return data
            
        except Exception as e:
            self.logger.error(f"Error downloading simulation {simulation_id}: {str(e)}")
            raise

    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch Einstein Toolkit simulation data"""
        self.logger.info("Fetching Einstein Toolkit simulation data")
        
        # Get list of available simulations
        simulations = self.fetch_simulation_metadata()
        
        # Download data for each simulation
        simulation_data = []
        for sim in simulations:
            try:
                data = self.download_simulation_data(sim['id'])
                data.update(sim)  # Add metadata to simulation data
                simulation_data.append(data)
            except Exception as e:
                self.logger.error(f"Failed to process simulation {sim['id']}: {str(e)}")
                continue
        
        return simulation_data
            
    def preprocess_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Preprocess Einstein Toolkit simulation data"""
        self.logger.info("Preprocessing Einstein Toolkit data")
        
        df = pd.DataFrame(data)
        
        # Convert dates
        if 'creation_date' in df.columns:
            df['creation_date'] = pd.to_datetime(df['creation_date'])
        
        # Add derived metrics
        if 'mass' in df.columns and 'horizon_mass' in df.columns:
            df['mass_loss_percentage'] = (
                (df['mass'] - df['horizon_mass']) / df['mass'] * 100
            ).where(df['horizon_mass'].notna())
        
        # Categorize simulations
        def categorize_simulation(row):
            if pd.isna(row['mass']) or pd.isna(row['spin']):
                return 'unknown'
            if row['mass'] > 50:
                return 'supermassive_black_hole'
            if row['spin'] > 0.9:
                return 'extreme_kerr'
            return 'standard'
        
        df['simulation_category'] = df.apply(categorize_simulation, axis=1)
        
        # Add metadata
        df['source'] = 'Einstein_Toolkit'
        df['scrape_timestamp'] = pd.Timestamp.now()
        
        # Calculate simulation complexity metrics
        if 'time_steps' in df.columns:
            df['simulation_complexity'] = pd.qcut(
                df['time_steps'],
                q=3,
                labels=['low', 'medium', 'high']
            )
        
        self.logger.info("Preprocessing completed")
        return df
