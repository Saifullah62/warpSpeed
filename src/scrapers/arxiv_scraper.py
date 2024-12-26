import requests
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

class ArXivScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.base_url = "http://export.arxiv.org/api/query"
        self.categories = [
            'gr-qc',          # General Relativity and Quantum Cosmology
            'hep-th',         # High Energy Physics - Theory
            'quant-ph',       # Quantum Physics
            'physics.space-ph', # Space Physics
            'physics.gen-ph',  # General Physics
            'physics.optics',  # Optics (for quantum vacuum effects)
            'astro-ph.HE',    # High Energy Astrophysical Phenomena
            'cond-mat.mtrl-sci' # Materials Science
        ]
        self.search_terms = [
            'warp drive',
            'alcubierre metric',
            'quantum vacuum',
            'negative energy density',
            'exotic matter',
            'casimir effect',
            'spacetime metric engineering',
            'faster than light',
            'ftl propulsion',
            'quantum tunneling',
            'quantum teleportation',
            'quantum entanglement propulsion',
            'metamaterials spacetime',
            'zero point energy',
            'quantum field theory propulsion'
        ]
        
    def fetch_papers(self, category: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Fetch papers from a specific ArXiv category"""
        # Combine category with relevant search terms
        search_query = f'cat:{category} AND ('
        search_query += ' OR '.join([f'all:"{term}"' for term in self.search_terms])
        search_query += ')'
        
        params = {
            'search_query': search_query,
            'max_results': max_results,
            'sortBy': 'lastUpdatedDate',
            'sortOrder': 'descending'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            
            papers = []
            for entry in root.findall('atom:entry', namespace):
                # Extract all categories
                categories = [cat.get('term') for cat in entry.findall('atom:category', namespace)]
                
                # Extract DOI if available
                doi = None
                for link in entry.findall('atom:link', namespace):
                    if link.get('title') == 'doi':
                        doi = link.get('href')
                
                paper = {
                    'title': entry.find('atom:title', namespace).text.strip(),
                    'authors': [author.find('atom:name', namespace).text for author in entry.findall('atom:author', namespace)],
                    'published': entry.find('atom:published', namespace).text,
                    'updated': entry.find('atom:updated', namespace).text,
                    'summary': entry.find('atom:summary', namespace).text.strip(),
                    'primary_category': category,
                    'all_categories': categories,
                    'arxiv_id': entry.find('atom:id', namespace).text.split('/')[-1],
                    'doi': doi,
                    'pdf_url': f"https://arxiv.org/pdf/{entry.find('atom:id', namespace).text.split('/')[-1]}.pdf"
                }
                papers.append(paper)
                
            return papers
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching arXiv papers for category {category}: {str(e)}")
            return []
            
    def fetch_data(self) -> List[Dict[str, Any]]:
        """Fetch papers from all relevant categories"""
        self.logger.info("Fetching ArXiv papers")
        
        all_papers = []
        for category in self.categories:
            self.logger.info(f"Fetching papers from category: {category}")
            papers = self.fetch_papers(category)
            all_papers.extend(papers)
            
        self.logger.info(f"Successfully fetched {len(all_papers)} papers")
        return all_papers
            
    def preprocess_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Preprocess ArXiv paper data"""
        self.logger.info("Preprocessing ArXiv data")
        
        df = pd.DataFrame(data)
        
        # Convert dates to datetime
        df['published'] = pd.to_datetime(df['published'])
        df['updated'] = pd.to_datetime(df['updated'])
        
        # Convert authors list to string
        df['authors'] = df['authors'].apply(lambda x: ', '.join(x))
        
        # Add metadata
        df['source'] = 'ArXiv'
        df['scrape_timestamp'] = pd.Timestamp.now()
        
        # Extract relevant keywords
        keywords = ['warp drive', 'spacetime', 'quantum field', 'casimir', 
                   'negative energy', 'exotic matter', 'alcubierre']
        
        def check_keywords(text):
            return [kw for kw in keywords if kw.lower() in text.lower()]
        
        df['relevant_keywords'] = df['summary'].apply(check_keywords)
        df['is_relevant'] = df['relevant_keywords'].apply(len) > 0
        
        self.logger.info("Preprocessing completed")
        return df
