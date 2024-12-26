import requests
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper
from bs4 import BeautifulSoup
import time

class PatentScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.base_urls = {
            'google_patents': 'https://patents.google.com',
            'uspto': 'https://patft.uspto.gov',
            'espacenet': 'https://worldwide.espacenet.com'
        }
        self.search_terms = [
            'propulsion system',
            'quantum propulsion',
            'spacetime manipulation',
            'field generator',
            'magnetic containment',
            'plasma containment',
            'energy field generation',
            'inertial dampening',
            'radiation shielding',
            'quantum vacuum engineering',
            'electromagnetic field control',
            'antimatter containment',
            'warp field technology',
            'exotic matter generation'
        ]
        
    def fetch_google_patents(self, search_term: str) -> List[Dict[str, Any]]:
        """Fetch patents from Google Patents"""
        patents = []
        url = f"{self.base_urls['google_patents']}/search"
        
        try:
            time.sleep(2)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            params = {
                'q': search_term,
                'num': 100
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for result in soup.find_all('article'):
                title = result.find('h3').text.strip() if result.find('h3') else ''
                abstract = result.find('div', class_='abstract').text.strip() if result.find('div', class_='abstract') else ''
                inventors = result.find('span', class_='inventors').text.strip() if result.find('span', class_='inventors') else ''
                date = result.find('time').text.strip() if result.find('time') else ''
                patent_number = result.find('span', class_='patent-number').text.strip() if result.find('span', class_='patent-number') else ''
                
                patents.append({
                    'title': title,
                    'abstract': abstract,
                    'inventors': inventors,
                    'date': date,
                    'patent_number': patent_number,
                    'url': f"{self.base_urls['google_patents']}/patent/{patent_number}",
                    'source': 'Google Patents'
                })
                
        except Exception as e:
            self.logger.error(f"Error fetching Google Patents data: {str(e)}")
            
        return patents

    def fetch_uspto(self, search_term: str) -> List[Dict[str, Any]]:
        """Fetch patents from USPTO"""
        patents = []
        url = f"{self.base_urls['uspto']}/netacgi/nph-Parser"
        
        try:
            time.sleep(2)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            params = {
                'Sect1': 'PTO2',
                'Sect2': 'HITOFF',
                'p': '1',
                'u': '/netahtml/PTO/search-bool.html',
                'r': '0',
                'f': 'S',
                'l': '50',
                'TERM1': search_term,
                'FIELD1': 'ABTX'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for row in soup.find_all('tr'):
                if not row.find_all('td'):
                    continue
                    
                cells = row.find_all('td')
                if len(cells) < 3:
                    continue
                    
                patent_number = cells[1].text.strip()
                title = cells[3].text.strip()
                
                patents.append({
                    'title': title,
                    'patent_number': patent_number,
                    'url': f"{self.base_urls['uspto']}/netacgi/nph-Parser?patentnumber={patent_number}",
                    'source': 'USPTO'
                })
                
        except Exception as e:
            self.logger.error(f"Error fetching USPTO data: {str(e)}")
            
        return patents

    def fetch_espacenet(self, search_term: str) -> List[Dict[str, Any]]:
        """Fetch patents from Espacenet"""
        patents = []
        url = f"{self.base_urls['espacenet']}/3.2/rest-services/search"
        
        try:
            time.sleep(2)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json'
            }
            
            params = {
                'q': search_term,
                'format': 'json',
                'Range': '1-100'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            data = response.json()
            
            for result in data.get('ops:world-patent-data', {}).get('ops:biblio-search', {}).get('ops:search-result', {}).get('ops:publication-reference', []):
                doc_info = result.get('document-id', {})
                
                patents.append({
                    'title': doc_info.get('invention-title', ''),
                    'patent_number': doc_info.get('doc-number', ''),
                    'date': doc_info.get('date', ''),
                    'applicant': doc_info.get('applicant', ''),
                    'url': f"{self.base_urls['espacenet']}/publicationDetails/biblio?CC=EP&NR={doc_info.get('doc-number', '')}",
                    'source': 'Espacenet'
                })
                
        except Exception as e:
            self.logger.error(f"Error fetching Espacenet data: {str(e)}")
            
        return patents

    def fetch_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all available patent data"""
        self.logger.info("Fetching patent data")
        
        all_data = {}
        for term in self.search_terms:
            results = []
            
            # Fetch from each source
            google_patents = self.fetch_google_patents(term)
            uspto_patents = self.fetch_uspto(term)
            espacenet_patents = self.fetch_espacenet(term)
            
            # Combine results
            results.extend(google_patents)
            results.extend(uspto_patents)
            results.extend(espacenet_patents)
            
            if results:
                all_data[term] = results
                
        return all_data

    def preprocess_data(self, data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess patent data"""
        self.logger.info("Preprocessing patent data")
        
        dfs = []
        
        for search_term, results in data.items():
            for result in results:
                df = pd.DataFrame([{
                    'title': result.get('title', ''),
                    'description': result.get('abstract', ''),
                    'inventors': result.get('inventors', ''),
                    'applicant': result.get('applicant', ''),
                    'date': result.get('date', ''),
                    'patent_number': result.get('patent_number', ''),
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
        
        # Remove duplicates based on patent number
        df = df.drop_duplicates(subset=['patent_number'])
        
        self.logger.info("Preprocessing completed")
        return df
