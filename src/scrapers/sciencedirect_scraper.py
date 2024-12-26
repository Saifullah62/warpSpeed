import requests
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper
from bs4 import BeautifulSoup
import time
import re

class ScienceDirectScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.base_url = "https://www.sciencedirect.com"
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
        
    def fetch_articles(self, search_term: str) -> List[Dict[str, Any]]:
        """Fetch articles related to a specific search term"""
        articles = []
        
        # Encode search term for URL
        encoded_term = requests.utils.quote(search_term)
        
        # Build search URL
        url = f"{self.base_url}/search"
        params = {
            'qs': encoded_term,
            'show': '100',  # Maximum results per page
            'sortBy': 'date'
        }
        
        try:
            # Add delay to be nice to the server
            time.sleep(2)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all article entries
            for article in soup.find_all('div', class_='result-item-content'):
                try:
                    # Extract title
                    title_elem = article.find('a', class_='result-list-title-link')
                    title = title_elem.text.strip() if title_elem else ''
                    article_url = title_elem['href'] if title_elem and title_elem.get('href') else ''
                    
                    # Extract authors
                    authors_elem = article.find('div', class_='authors')
                    authors = authors_elem.text.strip() if authors_elem else ''
                    
                    # Extract abstract
                    abstract_elem = article.find('div', class_='result-item-content')
                    abstract = abstract_elem.text.strip() if abstract_elem else ''
                    
                    # Extract journal
                    journal_elem = article.find('div', class_='publication-title')
                    journal = journal_elem.text.strip() if journal_elem else ''
                    
                    # Extract date
                    date_elem = article.find('div', class_='publication-year')
                    date = date_elem.text.strip() if date_elem else ''
                    
                    articles.append({
                        'title': title,
                        'authors': authors,
                        'abstract': abstract,
                        'journal': journal,
                        'date': date,
                        'url': f"{self.base_url}{article_url}" if article_url.startswith('/') else article_url
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error parsing article: {str(e)}")
                    continue
                    
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching articles for term '{search_term}': {str(e)}")
        
        return articles

    def fetch_article_content(self, url: str) -> str:
        """Fetch the full content of an article if available"""
        try:
            time.sleep(2)  # Be nice to the server
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the full text content if available
            content_elem = soup.find('div', class_='article-content')
            return content_elem.text.strip() if content_elem else ''
            
        except Exception as e:
            self.logger.error(f"Error fetching article content from {url}: {str(e)}")
            return ''

    def fetch_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all available ScienceDirect data"""
        self.logger.info("Fetching ScienceDirect Data")
        
        all_articles = {}
        for term in self.search_terms:
            articles = self.fetch_articles(term)
            if articles:
                # Fetch full content for each article
                for article in articles:
                    article['content'] = self.fetch_article_content(article['url'])
                all_articles[term] = articles
                
        return all_articles

    def preprocess_data(self, data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess ScienceDirect data"""
        self.logger.info("Preprocessing ScienceDirect data")
        
        dfs = []
        
        for search_term, articles in data.items():
            for article in articles:
                # Combine abstract and content for description
                description = article.get('content', '')
                if not description:
                    description = article.get('abstract', '')
                
                df = pd.DataFrame([{
                    'title': article.get('title', ''),
                    'description': description,
                    'authors': article.get('authors', ''),
                    'journal': article.get('journal', ''),
                    'date': article.get('date', ''),
                    'url': article.get('url', ''),
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
        df['source'] = 'ScienceDirect'
        df['scrape_timestamp'] = pd.Timestamp.now()
        
        # Remove duplicates based on URL
        df = df.drop_duplicates(subset=['url'])
        
        self.logger.info("Preprocessing completed")
        return df
