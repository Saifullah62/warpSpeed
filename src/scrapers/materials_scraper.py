import requests
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper
from bs4 import BeautifulSoup
import time

class MaterialsScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.base_urls = {
            'materials_project': 'https://materialsproject.org',
            'materials_cloud': 'https://materialscloud.org',
            'nomad': 'https://nomad-lab.eu'
        }
        self.search_terms = [
            'high energy materials',
            'quantum materials',
            'metamaterials',
            'exotic matter',
            'magnetic containment materials',
            'radiation shielding materials',
            'superconducting materials',
            'field generation materials',
            'plasma containment materials',
            'energy storage materials',
            'negative mass materials',
            'quantum vacuum materials',
            'spacetime engineering materials',
            'warp field materials'
        ]
        
    def fetch_materials_project(self, search_term: str) -> List[Dict[str, Any]]:
        """Fetch data from Materials Project"""
        results = []
        url = f"{self.base_urls['materials_project']}/api/v2/materials/search"
        
        try:
            time.sleep(2)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            
            params = {
                'q': search_term,
                'limit': 100
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for material in data.get('results', []):
                results.append({
                    'material_id': material.get('material_id', ''),
                    'formula': material.get('formula', ''),
                    'structure': material.get('structure', {}),
                    'properties': material.get('properties', {}),
                    'energy': material.get('energy', 0),
                    'band_gap': material.get('band_gap', 0),
                    'density': material.get('density', 0),
                    'source': 'Materials Project'
                })
                
        except Exception as e:
            self.logger.error(f"Error fetching Materials Project data: {str(e)}")
            
        return results

    def fetch_materials_cloud(self, search_term: str) -> List[Dict[str, Any]]:
        """Fetch data from Materials Cloud"""
        results = []
        url = f"{self.base_urls['materials_cloud']}/api/v2/search"
        
        try:
            time.sleep(2)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            
            params = {
                'query': search_term,
                'type': 'materials'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for entry in data.get('results', []):
                results.append({
                    'title': entry.get('title', ''),
                    'description': entry.get('description', ''),
                    'authors': entry.get('authors', []),
                    'publication_date': entry.get('publication_date', ''),
                    'doi': entry.get('doi', ''),
                    'url': entry.get('url', ''),
                    'keywords': entry.get('keywords', []),
                    'source': 'Materials Cloud'
                })
                
        except Exception as e:
            self.logger.error(f"Error fetching Materials Cloud data: {str(e)}")
            
        return results

    def fetch_nomad(self, search_term: str) -> List[Dict[str, Any]]:
        """Fetch data from NOMAD"""
        results = []
        url = f"{self.base_urls['nomad']}/api/v1/entries"
        
        try:
            time.sleep(2)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            
            params = {
                'q': search_term,
                'per_page': 100
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            for entry in data.get('results', []):
                results.append({
                    'entry_id': entry.get('entry_id', ''),
                    'material': entry.get('material', {}),
                    'method': entry.get('method', ''),
                    'properties': entry.get('properties', {}),
                    'authors': entry.get('authors', []),
                    'date': entry.get('upload_date', ''),
                    'doi': entry.get('doi', ''),
                    'source': 'NOMAD'
                })
                
        except Exception as e:
            self.logger.error(f"Error fetching NOMAD data: {str(e)}")
            
        return results

    def fetch_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all available materials data"""
        self.logger.info("Fetching materials data")
        
        all_data = {}
        for term in self.search_terms:
            results = []
            
            # Fetch from each source
            mp_results = self.fetch_materials_project(term)
            mc_results = self.fetch_materials_cloud(term)
            nomad_results = self.fetch_nomad(term)
            
            # Combine results
            results.extend(mp_results)
            results.extend(mc_results)
            results.extend(nomad_results)
            
            if results:
                all_data[term] = results
                
        return all_data

    def preprocess_data(self, data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess materials data"""
        self.logger.info("Preprocessing materials data")
        
        dfs = []
        
        for search_term, results in data.items():
            for result in results:
                # Convert complex objects to strings
                properties = str(result.get('properties', {}))
                structure = str(result.get('structure', {}))
                
                # Convert authors list to string if needed
                authors = result.get('authors', [])
                if isinstance(authors, list):
                    authors = ', '.join(authors)
                
                df = pd.DataFrame([{
                    'material_id': result.get('material_id', result.get('entry_id', '')),
                    'title': result.get('title', ''),
                    'description': result.get('description', ''),
                    'authors': authors,
                    'date': result.get('date', result.get('publication_date', '')),
                    'properties': properties,
                    'structure': structure,
                    'energy': result.get('energy', 0),
                    'band_gap': result.get('band_gap', 0),
                    'density': result.get('density', 0),
                    'doi': result.get('doi', ''),
                    'url': result.get('url', ''),
                    'source': result.get('source', ''),
                    'search_term': search_term
                }])
                
                dfs.append(df)
        
        # Combine all datasets
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.DataFrame()
        
        # Add metadata
        df['scrape_timestamp'] = pd.Timestamp.now()
        
        # Remove duplicates based on material_id
        df = df.drop_duplicates(subset=['material_id'])
        
        self.logger.info("Preprocessing completed")
        return df
