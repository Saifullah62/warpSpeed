import requests
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper
from bs4 import BeautifulSoup
import time
import re

class ADSScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.base_url = "https://ui.adsabs.harvard.edu"
        self.search_terms = [
            'warp drive',
            'alcubierre metric',
            'quantum vacuum propulsion',
            'negative energy density',
            'exotic matter propulsion',
            'casimir effect propulsion',
            'spacetime metric engineering',
            'faster than light propulsion',
            'quantum tunneling propulsion',
            'quantum field theory propulsion',
            'zero point energy propulsion',
            'quantum vacuum fluctuations',
            'dark energy propulsion',
            'antimatter propulsion',
            'quantum entanglement propulsion'
        ]
        self.journals = [
            'PhRvD',  # Physical Review D
            'PhRvL',  # Physical Review Letters
            'CQGra',  # Classical and Quantum Gravity
            'JHEP',   # Journal of High Energy Physics
            'Sci',    # Science
            'Natur',  # Nature
            'PhR',    # Physics Reports
            'RvMP'    # Reviews of Modern Physics
        ]
        
    def fetch_papers(self, search_term: str) -> List[Dict[str, Any]]:
        """Fetch papers related to a specific search term using web scraping"""
        papers = []
        
        # Encode search term for URL
        encoded_term = requests.utils.quote(f'"{search_term}"')
        journals = requests.utils.quote(' OR '.join(f'bibstem:{j}' for j in self.journals))
        
        # Build search URL
        url = f"{self.base_url}/search/q={encoded_term}"
        url += f"&fq={journals}"
        url += "&fq=property:refereed"  # Only peer-reviewed papers
        url += "&fq=year:2010-2024"    # Recent papers
        url += "&sort=date desc"
        url += "&p_=0"  # First page
        
        try:
            # Add delay to be nice to the server
            time.sleep(2)
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all paper entries
            for paper in soup.find_all('div', class_='s-results-item'):
                try:
                    # Extract title
                    title_elem = paper.find('a', class_='s-results-title')
                    title = title_elem.text.strip() if title_elem else ''
                    
                    # Extract authors
                    authors_elem = paper.find('div', class_='s-results-authors')
                    authors = authors_elem.text.strip() if authors_elem else ''
                    
                    # Extract abstract
                    abstract_elem = paper.find('div', class_='s-results-abstract')
                    abstract = abstract_elem.text.strip() if abstract_elem else ''
                    
                    # Extract date
                    date_elem = paper.find('div', class_='s-results-date')
                    date = date_elem.text.strip() if date_elem else ''
                    
                    # Extract bibcode
                    bibcode = ''
                    if title_elem and title_elem.get('href'):
                        bibcode = re.search(r'/abs/(\S+)', title_elem['href'])
                        bibcode = bibcode.group(1) if bibcode else ''
                    
                    # Extract journal
                    pub_elem = paper.find('div', class_='s-results-pub')
                    journal = pub_elem.text.strip() if pub_elem else ''
                    
                    papers.append({
                        'title': title,
                        'authors': authors,
                        'abstract': abstract,
                        'date': date,
                        'bibcode': bibcode,
                        'journal': journal,
                        'url': f"{self.base_url}/abs/{bibcode}" if bibcode else ''
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error parsing paper: {str(e)}")
                    continue
                    
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching papers for term '{search_term}': {str(e)}")
        
        return papers

    def fetch_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all available ADS data"""
        self.logger.info("Fetching ADS Data")
        
        all_papers = {}
        for term in self.search_terms:
            papers = self.fetch_papers(term)
            if papers:
                all_papers[term] = papers
                
        return all_papers

    def preprocess_data(self, data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess ADS data"""
        self.logger.info("Preprocessing ADS data")
        
        dfs = []
        
        for search_term, papers in data.items():
            for paper in papers:
                df = pd.DataFrame([{
                    'title': paper.get('title', ''),
                    'description': paper.get('abstract', ''),
                    'authors': paper.get('authors', ''),
                    'date': paper.get('date', ''),
                    'journal': paper.get('journal', ''),
                    'bibcode': paper.get('bibcode', ''),
                    'url': paper.get('url', ''),
                    'search_term': search_term,
                    'data_type': 'research_paper'
                }])
                
                dfs.append(df)
        
        # Combine all datasets
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.DataFrame()
        
        # Add metadata
        df['source'] = 'ADS'
        df['scrape_timestamp'] = pd.Timestamp.now()
        
        # Remove duplicates based on bibcode
        df = df.drop_duplicates(subset=['bibcode'])
        
        self.logger.info("Preprocessing completed")
        return df
