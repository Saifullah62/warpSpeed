import requests
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper
from bs4 import BeautifulSoup
import time

class AntimatterScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.base_urls = {
            'cern': 'https://home.cern',
            'fermilab': 'https://www.fnal.gov',
            'desy': 'https://www.desy.de'
        }
        self.search_terms = [
            'antimatter production',
            'antimatter containment',
            'matter antimatter reaction',
            'antimatter storage',
            'antimatter propulsion',
            'positron accumulation',
            'antiproton production',
            'antimatter catalyzed fusion',
            'antimatter energy conversion',
            'antimatter safety systems',
            'magnetic containment fields',
            'antimatter reactor design',
            'matter antimatter annihilation',
            'antimatter plasma dynamics'
        ]
        
    def fetch_cern_data(self, search_term: str) -> List[Dict[str, Any]]:
        """Fetch antimatter research data from CERN"""
        results = []
        url = f"{self.base_urls['cern']}/search"
        
        try:
            params = {
                'q': search_term,
                'category': 'antimatter'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for article in soup.find_all('article', class_='result-item'):
                title = article.find('h2').text.strip() if article.find('h2') else ''
                abstract = article.find('p', class_='description').text.strip() if article.find('p', class_='description') else ''
                date = article.find('time').text.strip() if article.find('time') else ''
                url = article.find('a')['href'] if article.find('a') else ''
                
                results.append({
                    'title': title,
                    'description': abstract,
                    'date': date,
                    'url': url,
                    'source': 'CERN',
                    'category': 'antimatter_research'
                })
                
        except Exception as e:
            self.logger.error(f"Error fetching CERN data: {str(e)}")
            
        return results

    def fetch_fermilab_data(self, search_term: str) -> List[Dict[str, Any]]:
        """Fetch antimatter research data from Fermilab"""
        results = []
        url = f"{self.base_urls['fermilab']}/search"
        
        try:
            params = {
                'q': search_term,
                'type': 'research'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for result in soup.find_all('div', class_='search-result'):
                title = result.find('h3').text.strip() if result.find('h3') else ''
                abstract = result.find('p').text.strip() if result.find('p') else ''
                date = result.find('span', class_='date').text.strip() if result.find('span', class_='date') else ''
                url = result.find('a')['href'] if result.find('a') else ''
                
                results.append({
                    'title': title,
                    'description': abstract,
                    'date': date,
                    'url': url,
                    'source': 'Fermilab',
                    'category': 'antimatter_research'
                })
                
        except Exception as e:
            self.logger.error(f"Error fetching Fermilab data: {str(e)}")
            
        return results

    def fetch_desy_data(self, search_term: str) -> List[Dict[str, Any]]:
        """Fetch antimatter research data from DESY"""
        results = []
        url = f"{self.base_urls['desy']}/search"
        
        try:
            params = {
                'q': search_term,
                'section': 'research'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for result in soup.find_all('div', class_='search-item'):
                title = result.find('h4').text.strip() if result.find('h4') else ''
                abstract = result.find('div', class_='abstract').text.strip() if result.find('div', class_='abstract') else ''
                date = result.find('span', class_='date').text.strip() if result.find('span', class_='date') else ''
                url = result.find('a')['href'] if result.find('a') else ''
                
                results.append({
                    'title': title,
                    'description': abstract,
                    'date': date,
                    'url': url,
                    'source': 'DESY',
                    'category': 'antimatter_research'
                })
                
        except Exception as e:
            self.logger.error(f"Error fetching DESY data: {str(e)}")
            
        return results

    def fetch_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all available antimatter research data"""
        self.logger.info("Fetching antimatter research data")
        
        all_results = {}
        for term in self.search_terms:
            results = []
            
            # Fetch from each source
            cern_results = self.fetch_cern_data(term)
            fermilab_results = self.fetch_fermilab_data(term)
            desy_results = self.fetch_desy_data(term)
            
            # Combine results
            results.extend(cern_results)
            results.extend(fermilab_results)
            results.extend(desy_results)
            
            if results:
                all_results[term] = results
                
        return all_results

    def preprocess_data(self, data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess antimatter research data"""
        self.logger.info("Preprocessing antimatter research data")
        
        dfs = []
        
        for search_term, results in data.items():
            for result in results:
                df = pd.DataFrame([{
                    'title': result.get('title', ''),
                    'description': result.get('description', ''),
                    'date': result.get('date', ''),
                    'url': result.get('url', ''),
                    'source': result.get('source', ''),
                    'category': result.get('category', ''),
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
        
        # Remove duplicates based on URL
        df = df.drop_duplicates(subset=['url'])
        
        self.logger.info("Preprocessing completed")
        return df
