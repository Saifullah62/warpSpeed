import requests
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper
import os

class NASAScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        # Hardcoded API key for private implementation
        self.api_key = "PBI6QS2eOkQigQpGPHNNCTxvPbSzakdW2Nq8R7yS"
        self.base_url = "https://api.nasa.gov"
        
    def fetch_exoplanet_data(self) -> List[Dict[str, Any]]:
        """Fetch space propulsion and quantum research data"""
        endpoint = f"{self.base_url}/planetary/apod"
        params = {
            'api_key': self.api_key,
            'count': 50  # Fetch 50 entries
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching NASA APOD data: {str(e)}")
            raise

    def fetch_star_data(self) -> List[Dict[str, Any]]:
        """Fetch NEO data for potential propulsion research"""
        endpoint = f"{self.base_url}/neo/rest/v1/neo/browse"
        params = {
            'api_key': self.api_key
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            return response.json()['near_earth_objects']
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching NASA NEO data: {str(e)}")
            raise

    def fetch_techport_data(self) -> List[Dict[str, Any]]:
        """Fetch NASA TechPort data for advanced propulsion research"""
        endpoint = "https://techport.nasa.gov/api/projects/search"
        params = {
            'searchQuery': 'propulsion OR quantum OR warp OR antimatter',
            'updatedSince': '2020-01-01'
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            projects = response.json().get('projects', [])
            
            # Get detailed info for each project
            detailed_projects = []
            for project in projects[:20]:  # Limit to 20 projects to avoid too many requests
                try:
                    detail_response = requests.get(f"https://techport.nasa.gov/api/projects/{project['projectId']}", timeout=10)
                    detail_response.raise_for_status()
                    detailed_projects.append(detail_response.json())
                except Exception as e:
                    self.logger.error(f"Error fetching project details: {str(e)}")
                    continue
            
            return detailed_projects
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching NASA TechPort data: {str(e)}")
            return []

    def fetch_ntrs_data(self) -> List[Dict[str, Any]]:
        """Fetch NASA Technical Reports Server data"""
        endpoint = "https://ntrs.nasa.gov/api/citations/search"
        params = {
            'q': '(propulsion OR quantum OR warp OR antimatter) AND year:[2020 TO 2024]',
            'size': 100
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            return response.json().get('results', [])
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching NASA NTRS data: {str(e)}")
            return []

    def fetch_images_data(self) -> List[Dict[str, Any]]:
        """Fetch NASA Image Library data"""
        endpoint = "https://images-api.nasa.gov/search"
        params = {
            'q': 'propulsion quantum warp drive engine',
            'media_type': 'image',
            'year_start': '2020'
        }
        
        try:
            response = requests.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            return response.json().get('collection', {}).get('items', [])
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching NASA Images data: {str(e)}")
            return []

    def fetch_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all available NASA data"""
        self.logger.info("Fetching NASA data")
        
        return {
            'techport': self.fetch_techport_data(),
            'ntrs': self.fetch_ntrs_data(),
            'images': self.fetch_images_data(),
            'apod': self.fetch_exoplanet_data(),
            'neo': self.fetch_star_data()
        }

    def preprocess_data(self, data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess NASA data"""
        self.logger.info("Preprocessing NASA data")
        
        dfs = []
        
        # Process TechPort data
        if data['techport']:
            techport_df = pd.DataFrame(data['techport'])
            techport_df['data_type'] = 'research_project'
            techport_df['title'] = techport_df.get('title', '')
            techport_df['description'] = techport_df.apply(
                lambda x: f"{x.get('description', '')} Benefits: {x.get('benefits', '')}", axis=1
            )
            techport_df['date'] = techport_df.get('lastUpdated', '')
            dfs.append(techport_df)
        
        # Process NTRS data
        if data['ntrs']:
            ntrs_df = pd.DataFrame(data['ntrs'])
            ntrs_df['data_type'] = 'technical_report'
            ntrs_df['title'] = ntrs_df.get('title', '')
            ntrs_df['description'] = ntrs_df.get('abstract', '')
            ntrs_df['date'] = ntrs_df.get('publicationDate', '')
            dfs.append(ntrs_df)
        
        # Process Images data
        if data['images']:
            images_df = pd.DataFrame([item['data'][0] for item in data['images'] if item.get('data')])
            images_df['data_type'] = 'image'
            images_df['title'] = images_df.get('title', '')
            images_df['description'] = images_df.get('description', '')
            images_df['date'] = images_df.get('date_created', '')
            dfs.append(images_df)
        
        # Process APOD data
        if data['apod']:
            apod_df = pd.DataFrame(data['apod'])
            apod_df['data_type'] = 'astronomy'
            apod_df['title'] = apod_df.get('title', '')
            apod_df['description'] = apod_df.get('explanation', '')
            apod_df['date'] = apod_df.get('date', '')
            dfs.append(apod_df)
        
        # Process NEO data
        if data['neo']:
            neo_df = pd.DataFrame(data['neo'])
            neo_df['data_type'] = 'neo'
            neo_df['title'] = neo_df.get('name', '')
            neo_df['description'] = neo_df.apply(
                lambda x: f"NEO with diameter between {x.get('estimated_diameter', {}).get('kilometers', {}).get('estimated_diameter_min', 0):.2f} and {x.get('estimated_diameter', {}).get('kilometers', {}).get('estimated_diameter_max', 0):.2f} km", 
                axis=1
            )
            neo_df['date'] = neo_df.get('orbital_data', {}).get('first_observation_date', '')
            dfs.append(neo_df)
        
        # Combine all datasets
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.DataFrame()
        
        # Add metadata
        df['source'] = 'NASA'
        df['scrape_timestamp'] = pd.Timestamp.now()
        
        # Select and rename columns
        df = df[['title', 'description', 'date', 'data_type', 'source', 'scrape_timestamp']]
        
        self.logger.info("Preprocessing completed")
        return df
