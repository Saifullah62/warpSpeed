import requests
import pandas as pd
from typing import Dict, List, Any
from .base_scraper import BaseScraper
from bs4 import BeautifulSoup
import time
import logging

class AdvancedConceptsScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        
        # Research areas and their associated keywords
        self.research_areas = {
            'zero_point_energy': {
                'keywords': [
                    'zero-point energy',
                    'quantum vacuum energy',
                    'vacuum fluctuations',
                    'casimir effect',
                    'quantum energy extraction',
                    'vacuum energy density',
                    'quantum field vacuum',
                    'zero-point field',
                    'quantum vacuum engineering',
                    'vacuum energy harvesting'
                ],
                'sources': [
                    'https://arxiv.org/search/',
                    'https://www.sciencedirect.com/search',
                    'https://physics.aps.org/',
                    'https://www.nature.com/subjects/quantum-physics'
                ]
            },
            'spacetime_manipulation': {
                'keywords': [
                    'spacetime manipulation',
                    'metric engineering',
                    'non-exotic matter',
                    'spacetime curvature',
                    'warp field generation',
                    'alternative warp drive',
                    'spacetime modification',
                    'gravitational engineering',
                    'metric tensor manipulation',
                    'spacetime geometry control'
                ],
                'sources': [
                    'https://arxiv.org/search/',
                    'https://journals.aps.org/search',
                    'https://www.worldscientific.com/search'
                ]
            },
            'integrated_systems': {
                'keywords': [
                    'warp field navigation',
                    'bubble shielding',
                    'integrated warp systems',
                    'navigation field integration',
                    'warp bubble properties',
                    'field-based shielding',
                    'unified warp systems',
                    'bubble field properties',
                    'warp navigation control',
                    'field harmonization'
                ],
                'sources': [
                    'https://ieeexplore.ieee.org/search',
                    'https://www.sciencedirect.com/search',
                    'https://arc.aiaa.org/search'
                ]
            }
        }
        
        self.source_handlers = {
            'arxiv': self._scrape_arxiv,
            'sciencedirect': self._scrape_sciencedirect,
            'physics_aps': self._scrape_aps,
            'nature': self._scrape_nature,
            'ieee': self._scrape_ieee,
            'aiaa': self._scrape_aiaa
        }

    def _scrape_arxiv(self, keyword: str) -> List[Dict[str, Any]]:
        """Scrape arXiv papers"""
        results = []
        base_url = "https://export.arxiv.org/api/query"
        
        try:
            params = {
                'search_query': f'all:{keyword}',
                'start': 0,
                'max_results': 100,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            soup = BeautifulSoup(response.content, 'xml')
            
            for entry in soup.find_all('entry'):
                results.append({
                    'title': entry.title.text,
                    'authors': [author.text for author in entry.find_all('author')],
                    'summary': entry.summary.text,
                    'published': entry.published.text,
                    'url': entry.id.text,
                    'source': 'arXiv'
                })
                
        except Exception as e:
            self.logger.error(f"Error scraping arXiv for {keyword}: {str(e)}")
            
        return results

    def _scrape_sciencedirect(self, keyword: str) -> List[Dict[str, Any]]:
        """Scrape ScienceDirect papers"""
        results = []
        base_url = "https://api.elsevier.com/content/search/sciencedirect"
        
        try:
            headers = {
                'Accept': 'application/json',
                'User-Agent': 'Mozilla/5.0'
            }
            
            params = {
                'query': keyword,
                'count': 100,
                'sort': 'relevance'
            }
            
            response = requests.get(base_url, headers=headers, params=params, timeout=30)
            data = response.json()
            
            for entry in data.get('search-results', {}).get('entry', []):
                results.append({
                    'title': entry.get('dc:title'),
                    'authors': entry.get('dc:creator', '').split(';'),
                    'summary': entry.get('dc:description', ''),
                    'published': entry.get('prism:coverDate'),
                    'url': entry.get('prism:url'),
                    'source': 'ScienceDirect'
                })
                
        except Exception as e:
            self.logger.error(f"Error scraping ScienceDirect for {keyword}: {str(e)}")
            
        return results

    def _scrape_aps(self, keyword: str) -> List[Dict[str, Any]]:
        """Scrape APS Physics papers"""
        results = []
        base_url = "https://journals.aps.org/search/results"
        
        try:
            params = {
                'q': keyword,
                'sort': 'relevance',
                'per_page': 100
            }
            
            response = requests.get(base_url, params=params, timeout=30)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for article in soup.find_all('article'):
                results.append({
                    'title': article.find('h3').text.strip(),
                    'authors': [a.text for a in article.find_all('a', class_='authors')],
                    'summary': article.find('p', class_='abstract').text.strip(),
                    'published': article.find('time').text.strip(),
                    'url': article.find('a', class_='title')['href'],
                    'source': 'APS Physics'
                })
                
        except Exception as e:
            self.logger.error(f"Error scraping APS Physics for {keyword}: {str(e)}")
            
        return results

    def fetch_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch data for all research areas"""
        all_data = {}
        
        for area, details in self.research_areas.items():
            area_results = []
            self.logger.info(f"Fetching data for {area}")
            
            for keyword in details['keywords']:
                # Add delay between requests
                time.sleep(2)
                
                # Fetch from each source
                for source in details['sources']:
                    if 'arxiv.org' in source:
                        results = self._scrape_arxiv(keyword)
                    elif 'sciencedirect.com' in source:
                        results = self._scrape_sciencedirect(keyword)
                    elif 'physics.aps.org' in source:
                        results = self._scrape_aps(keyword)
                    # Add more source handlers as needed
                    
                    area_results.extend(results)
            
            all_data[area] = area_results
            
        return all_data

    def preprocess_data(self, data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess the scraped data"""
        dfs = []
        
        for area, results in data.items():
            for result in results:
                # Convert authors list to string
                authors = result.get('authors', [])
                if isinstance(authors, list):
                    authors = ', '.join(authors)
                
                df = pd.DataFrame([{
                    'title': result.get('title', ''),
                    'authors': authors,
                    'summary': result.get('summary', ''),
                    'published': result.get('published', ''),
                    'url': result.get('url', ''),
                    'source': result.get('source', ''),
                    'research_area': area
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
        
        return df
