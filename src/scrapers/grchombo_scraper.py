import requests
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper
import h5py
import numpy as np
import tempfile
import os

class GRChomboScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.base_url = "https://www.grchombo.org/data"
        
    def fetch_simulation_catalog(self) -> List[Dict[str, Any]]:
        """Fetch available GRChombo simulations"""
        try:
            response = requests.get(f"{self.base_url}/simulations/catalog.json")
            response.raise_for_status()
            return response.json()['simulations']
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching GRChombo catalog: {str(e)}")
            raise

    def download_simulation(self, sim_id: str) -> Dict[str, Any]:
        """Download and process a specific simulation"""
        try:
            # Download simulation data
            response = requests.get(f"{self.base_url}/simulations/{sim_id}/data.h5")
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                tmp_file.write(response.content)
                tmp_path = tmp_file.name
            
            # Extract data from HDF5 file
            with h5py.File(tmp_path, 'r') as f:
                data = {
                    'simulation_id': sim_id,
                    'metric_components': {
                        key: f[f'metric/{key}'][()]
                        for key in f['metric'].keys()
                    },
                    'matter_fields': {
                        key: f[f'matter/{key}'][()]
                        for key in f['matter'].keys()
                    } if 'matter' in f else {},
                    'parameters': {
                        key: f[f'parameters/{key}'][()]
                        for key in f['parameters'].keys()
                    }
                }
                
                # Calculate key metrics
                if 'metric/g_tt' in f:  # time-time component of metric
                    data['warp_factor'] = np.min(f['metric/g_tt'][()])
                if 'matter/energy_density' in f:
                    data['min_energy_density'] = np.min(f['matter/energy_density'][()])
                    data['max_energy_density'] = np.max(f['matter/energy_density'][()])
            
            # Clean up
            os.unlink(tmp_path)
            return data
            
        except Exception as e:
            self.logger.error(f"Error processing simulation {sim_id}: {str(e)}")
            raise

    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch GRChombo simulation data"""
        self.logger.info("Fetching GRChombo simulation data")
        
        simulations = []
        catalog = self.fetch_simulation_catalog()
        
        for sim in catalog:
            try:
                sim_data = self.download_simulation(sim['id'])
                sim_data.update(sim)  # Add catalog metadata
                simulations.append(sim_data)
            except Exception as e:
                self.logger.error(f"Failed to process simulation {sim['id']}: {str(e)}")
                continue
        
        return simulations
            
    def preprocess_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Preprocess GRChombo simulation data"""
        self.logger.info("Preprocessing GRChombo data")
        
        # Extract key metrics into flat structure
        processed_data = []
        for sim in data:
            record = {
                'simulation_id': sim['simulation_id'],
                'warp_factor': sim.get('warp_factor'),
                'min_energy_density': sim.get('min_energy_density'),
                'max_energy_density': sim.get('max_energy_density'),
                'has_exotic_matter': sim.get('min_energy_density', 0) < 0,
                'metric_type': sim.get('parameters', {}).get('metric_type'),
                'simulation_time': sim.get('parameters', {}).get('simulation_time'),
                'resolution': sim.get('parameters', {}).get('resolution')
            }
            processed_data.append(record)
        
        df = pd.DataFrame(processed_data)
        
        # Add metadata
        df['source'] = 'GRChombo'
        df['scrape_timestamp'] = pd.Timestamp.now()
        
        # Categorize simulations
        def categorize_simulation(row):
            if pd.isna(row['warp_factor']):
                return 'unknown'
            if row['warp_factor'] < -1:  # Significant time dilation
                return 'extreme_spacetime'
            if row['has_exotic_matter']:
                return 'exotic_matter'
            return 'standard'
        
        df['simulation_category'] = df.apply(categorize_simulation, axis=1)
        
        self.logger.info("Preprocessing completed")
        return df
