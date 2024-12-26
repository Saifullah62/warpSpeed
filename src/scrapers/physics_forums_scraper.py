import requests
import pandas as pd
from typing import Dict, Any, List
from .base_scraper import BaseScraper
from bs4 import BeautifulSoup
import time
import re

class PhysicsForumsScraper(BaseScraper):
    def __init__(self, output_dir: str):
        super().__init__(output_dir)
        self.base_url = "https://www.physicsforums.com"
        self.search_terms = [
            'warp drive',
            'alcubierre metric',
            'quantum vacuum propulsion',
            'negative energy density',
            'exotic matter',
            'casimir effect propulsion',
            'spacetime metric engineering',
            'faster than light',
            'quantum tunneling propulsion',
            'quantum field theory propulsion'
        ]
        self.subforums = [
            'relativity',
            'quantum-physics',
            'beyond-the-standard-model',
            'special-general-relativity',
            'quantum-mechanics'
        ]
        
    def fetch_threads(self, search_term: str) -> List[Dict[str, Any]]:
        """Fetch threads related to a specific search term"""
        threads = []
        
        # Encode search term for URL
        encoded_term = requests.utils.quote(search_term)
        
        # Build search URL
        url = f"{self.base_url}/search/search-results?q={encoded_term}"
        
        try:
            # Add delay to be nice to the server
            time.sleep(2)
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all thread entries
            for thread in soup.find_all('div', class_='search-result'):
                try:
                    # Extract title
                    title_elem = thread.find('a', class_='search-result-title')
                    title = title_elem.text.strip() if title_elem else ''
                    thread_url = title_elem['href'] if title_elem and title_elem.get('href') else ''
                    
                    # Extract preview text
                    preview_elem = thread.find('div', class_='search-result-preview')
                    preview = preview_elem.text.strip() if preview_elem else ''
                    
                    # Extract forum section
                    forum_elem = thread.find('a', class_='search-result-forum')
                    forum = forum_elem.text.strip() if forum_elem else ''
                    
                    # Extract date
                    date_elem = thread.find('time')
                    date = date_elem['datetime'] if date_elem and date_elem.get('datetime') else ''
                    
                    # Only include if it's from a relevant subforum
                    if any(subforum in forum.lower() for subforum in self.subforums):
                        threads.append({
                            'title': title,
                            'preview': preview,
                            'forum': forum,
                            'date': date,
                            'url': f"{self.base_url}{thread_url}" if thread_url.startswith('/') else thread_url
                        })
                    
                except Exception as e:
                    self.logger.error(f"Error parsing thread: {str(e)}")
                    continue
                    
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching threads for term '{search_term}': {str(e)}")
        
        return threads

    def fetch_thread_content(self, url: str) -> str:
        """Fetch the full content of a thread"""
        try:
            time.sleep(2)  # Be nice to the server
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the main post content
            content_elem = soup.find('div', class_='message-content')
            return content_elem.text.strip() if content_elem else ''
            
        except Exception as e:
            self.logger.error(f"Error fetching thread content from {url}: {str(e)}")
            return ''

    def fetch_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch all available Physics Forums data"""
        self.logger.info("Fetching Physics Forums Data")
        
        all_threads = {}
        for term in self.search_terms:
            threads = self.fetch_threads(term)
            if threads:
                # Fetch full content for each thread
                for thread in threads:
                    thread['content'] = self.fetch_thread_content(thread['url'])
                all_threads[term] = threads
                
        return all_threads

    def preprocess_data(self, data: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """Preprocess Physics Forums data"""
        self.logger.info("Preprocessing Physics Forums data")
        
        dfs = []
        
        for search_term, threads in data.items():
            for thread in threads:
                # Combine preview and content for description
                description = thread.get('content', '')
                if not description:
                    description = thread.get('preview', '')
                
                df = pd.DataFrame([{
                    'title': thread.get('title', ''),
                    'description': description,
                    'forum': thread.get('forum', ''),
                    'date': thread.get('date', ''),
                    'url': thread.get('url', ''),
                    'search_term': search_term,
                    'data_type': 'forum_thread'
                }])
                
                dfs.append(df)
        
        # Combine all datasets
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.DataFrame()
        
        # Add metadata
        df['source'] = 'Physics Forums'
        df['scrape_timestamp'] = pd.Timestamp.now()
        
        # Remove duplicates based on URL
        df = df.drop_duplicates(subset=['url'])
        
        self.logger.info("Preprocessing completed")
        return df
