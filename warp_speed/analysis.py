"""
Analysis tools for the Warp Speed Dataset.
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class Analysis:
    """Analyze research papers and extract insights."""
    
    def __init__(self):
        """Initialize the analysis tools."""
        self._setup_logging()
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.lda = LatentDirichletAllocation(n_components=10, random_state=42)
    
    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def build_citation_network(self, papers: List[Dict]) -> nx.DiGraph:
        """
        Build a citation network from papers.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            NetworkX DiGraph representing the citation network
        """
        G = nx.DiGraph()
        
        # Add nodes and edges
        for paper in papers:
            G.add_node(paper['id'], 
                      title=paper['title'],
                      category=paper['category'])
            
            if 'references' in paper:
                for ref in paper['references']:
                    G.add_edge(paper['id'], ref)
        
        logger.info(f"Built citation network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def analyze_topics(self, papers: List[Dict], n_topics: int = 10) -> Tuple[List[List[str]], np.ndarray]:
        """
        Perform topic modeling on papers.
        
        Args:
            papers: List of paper dictionaries
            n_topics: Number of topics to extract
            
        Returns:
            Tuple of (list of top words per topic, document-topic matrix)
        """
        # Prepare text data
        texts = [p['abstract'] for p in papers]
        X = self.vectorizer.fit_transform(texts)
        
        # Fit LDA model
        self.lda.n_components = n_topics
        doc_topic_matrix = self.lda.fit_transform(X)
        
        # Get top words per topic
        feature_names = self.vectorizer.get_feature_names()
        topics = []
        for topic in self.lda.components_:
            top_words = [feature_names[i] for i in topic.argsort()[:-10:-1]]
            topics.append(top_words)
        
        logger.info(f"Extracted {n_topics} topics from {len(papers)} papers")
        return topics, doc_topic_matrix
    
    def find_connections(self, papers: List[Dict]) -> Dict:
        """
        Find connections between papers across categories.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Dictionary of connection statistics
        """
        # Create category pairs
        categories = set(p['category'] for p in papers)
        connections = {}
        
        for cat1 in categories:
            for cat2 in categories:
                if cat1 != cat2:
                    papers1 = [p for p in papers if p['category'] == cat1]
                    papers2 = [p for p in papers if p['category'] == cat2]
                    
                    # Count references between categories
                    ref_count = 0
                    for p1 in papers1:
                        if 'references' in p1:
                            p2_ids = set(p['id'] for p in papers2)
                            ref_count += len(set(p1['references']) & p2_ids)
                    
                    if ref_count > 0:
                        connections[(cat1, cat2)] = ref_count
        
        logger.info(f"Found {len(connections)} cross-category connections")
        return connections
    
    def analyze_trends(self, papers: List[Dict]) -> pd.DataFrame:
        """
        Analyze research trends over time.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            DataFrame with trend analysis
        """
        # Convert to DataFrame
        df = pd.DataFrame(papers)
        
        # Ensure datetime
        df['publication_date'] = pd.to_datetime(df['publication_date'])
        
        # Group by date and category
        trends = df.groupby([
            pd.Grouper(key='publication_date', freq='M'),
            'category'
        ]).size().unstack(fill_value=0)
        
        logger.info(f"Analyzed trends across {len(trends)} time periods")
        return trends
    
    def generate_report(self, papers: List[Dict]) -> Dict:
        """
        Generate a comprehensive analysis report.
        
        Args:
            papers: List of paper dictionaries
            
        Returns:
            Dictionary containing various analysis results
        """
        report = {}
        
        # Basic statistics
        report['total_papers'] = len(papers)
        report['categories'] = dict(pd.Series([p['category'] for p in papers]).value_counts())
        
        # Citation network
        G = self.build_citation_network(papers)
        report['network_stats'] = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'density': nx.density(G),
            'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes()
        }
        
        # Topic modeling
        topics, _ = self.analyze_topics(papers)
        report['topics'] = topics
        
        # Trends
        report['trends'] = self.analyze_trends(papers).to_dict()
        
        # Cross-category connections
        report['connections'] = self.find_connections(papers)
        
        logger.info("Generated comprehensive analysis report")
        return report
