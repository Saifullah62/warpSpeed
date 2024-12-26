import requests
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper
from bs4 import BeautifulSoup
import time
import re

class ResearchGateScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.base_url = "https://www.researchgate.net"
        self.search_terms = [
            'warp drive physics',
            'alcubierre metric',
            'quantum vacuum propulsion',
            'negative energy density spacetime',
            'exotic matter warp drive',
            'casimir effect propulsion',
            'spacetime metric engineering',
            'faster than light propulsion',
            'quantum tunneling propulsion',
            'quantum field theory propulsion',
            'zero point energy propulsion',
            'quantum vacuum fluctuations',
            'dark energy propulsion',
            'metamaterial spacetime'
        ]
        
    def fetch_publications(self, search_term: str) -> List[Dict[str, Any]]:
        """Fetch publications related to a specific search term"""
        publications = []
        
        # Encode search term for URL
        encoded_term = requests.utils.quote(search_term)
        
        # Build search URL
        url = f"{self.base_url}/search/publication?q={encoded_term}"
        
        try:
            # Add delay to be nice to the server
            time.sleep(2)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all publication entries
            for pub in soup.find_all('div', class_='nova-legacy-v-publication-item__stack'):
                try:
                    # Extract title
                    title_elem = pub.find('a', class_='nova-legacy-v-publication-item__title')
                    title = title_elem.text.strip() if title_elem else ''
                    pub_url = title_elem['href'] if title_elem and title_elem.get('href') else ''
                    
                    # Extract authors
                    authors_elem = pub.find('div', class_='nova-legacy-v-publication-item__authors')
                    authors = authors_elem.text.strip() if authors_elem else ''
                    
                    # Extract abstract/preview
                    preview_elem = pub.find('div', class_='nova-legacy-v-publication-item__description')
                    preview = preview_elem.text.strip() if preview_elem else ''
                    
                    # Extract journal/conference
                    venue_elem = pub.find('div', class_='nova-legacy-v-publication-item__meta-info')
                    venue = venue_elem.text.strip() if venue_elem else ''
                    
                    # Extract date
                    date_elem = pub.find('div', class_='nova-legacy-v-publication-item__meta-data')
                    date = date_elem.text.strip() if date_elem else ''
                    
                    # Extract citation count
                    citations_elem = pub.find('div', class_='nova-legacy-v-publication-item__metrics')
                    citations = citations_elem.text.strip() if citations_elem else ''
                    
                    publications.append({
                        'title': title,
                        'authors': authors,
                        'preview': preview,
                        'venue': venue,
                        'date': date,
                        'citations': citations,
                        'url': f"{self.base_url}{pub_url}" if pub_url.startswith('/') else pub_url
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error parsing publication: {str(e)}")
                    continue
                    
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching publications for term '{search_term}': {str(e)}")
        
        return publications

    def fetch_publication_content(self, url: str) -> str:
        """Fetch the full content of a publication if available"""
        try:
            time.sleep(2)  # Be nice to the server
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the full text content if available
            content_elem = soup.find('div', class_='nova-legacy-e-text--spacing-auto')
            return content_elem.text.strip() if content_elem else ''
            
        except Exception as e:
            self.logger.error(f"Error fetching publication content from {url}: {str(e)}")
            return ''

    def fetch_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all available ResearchGate data"""
        self.logger.info("Fetching ResearchGate Data")
        
        all_publications = {}
        for term in self.search_terms:
            publications = self.fetch_publications(term)
            if publications:
                # Fetch full content for each publication
                for pub in publications:
                    pub['content'] = self.fetch_publication_content(pub['url'])
                all_publications[term] = publications
                
        return all_publications

    def preprocess_data(self, data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess ResearchGate data"""
        self.logger.info("Preprocessing ResearchGate data")
        
        dfs = []
        
        for search_term, publications in data.items():
            for pub in publications:
                # Combine preview and content for description
                description = pub.get('content', '')
                if not description:
                    description = pub.get('preview', '')
                
                # Extract citation count number
                citations = pub.get('citations', '')
                citation_count = re.search(r'\d+', citations)
                citation_count = int(citation_count.group()) if citation_count else 0
                
                df = pd.DataFrame([{
                    'title': pub.get('title', ''),
                    'description': description,
                    'authors': pub.get('authors', ''),
                    'venue': pub.get('venue', ''),
                    'date': pub.get('date', ''),
                    'citations': citation_count,
                    'url': pub.get('url', ''),
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
        df['source'] = 'ResearchGate'
        df['scrape_timestamp'] = pd.Timestamp.now()
        
        # Remove duplicates based on URL
        df = df.drop_duplicates(subset=['url'])
        
        self.logger.info("Preprocessing completed")
        return df
