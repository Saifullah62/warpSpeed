"""
Visualization tools for the Warp Speed Dataset.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    """Create visualizations for research paper analysis."""
    
    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / 'visualizations'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        self._set_style()
    
    def _setup_logging(self):
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _set_style(self):
        """Set default visualization style."""
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_category_distribution(self, papers: List[Dict], save: bool = True) -> None:
        """
        Plot distribution of papers across categories.
        
        Args:
            papers: List of paper dictionaries
            save: Whether to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Count papers per category
        categories = pd.Series([p['category'] for p in papers]).value_counts()
        
        # Create bar plot
        sns.barplot(x=categories.values, y=categories.index)
        plt.title('Distribution of Papers by Category')
        plt.xlabel('Number of Papers')
        plt.ylabel('Category')
        
        if save:
            plt.savefig(self.output_dir / 'category_distribution.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved category distribution plot to {self.output_dir}")
        
        plt.close()
    
    def plot_citation_network(self, G: nx.DiGraph, save: bool = True) -> None:
        """
        Plot citation network.
        
        Args:
            G: NetworkX DiGraph
            save: Whether to save the plot
        """
        plt.figure(figsize=(15, 15))
        
        # Calculate node sizes based on in-degree
        node_size = [G.in_degree(n) * 100 + 100 for n in G.nodes()]
        
        # Calculate node colors based on category
        categories = nx.get_node_attributes(G, 'category')
        unique_categories = list(set(categories.values()))
        color_map = {cat: i for i, cat in enumerate(unique_categories)}
        node_colors = [color_map[categories[n]] for n in G.nodes()]
        
        # Draw network
        pos = nx.spring_layout(G, k=1, iterations=50)
        nx.draw(G, pos,
               node_size=node_size,
               node_color=node_colors,
               with_labels=False,
               alpha=0.7)
        
        # Add legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                    markerfacecolor=plt.cm.viridis(color_map[cat]/len(color_map)),
                                    label=cat, markersize=10)
                         for cat in unique_categories]
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        if save:
            plt.savefig(self.output_dir / 'citation_network.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved citation network plot to {self.output_dir}")
        
        plt.close()
    
    def plot_topic_distribution(self, topics: List[List[str]], doc_topics: np.ndarray,
                              save: bool = True) -> None:
        """
        Plot topic distribution.
        
        Args:
            topics: List of top words per topic
            doc_topics: Document-topic matrix
            save: Whether to save the plot
        """
        plt.figure(figsize=(15, 10))
        
        # Plot topic distribution
        topic_shares = doc_topics.mean(axis=0)
        topic_names = [f"Topic {i+1}\n({', '.join(words[:3])})" 
                      for i, words in enumerate(topics)]
        
        sns.barplot(x=range(len(topic_shares)), y=topic_shares)
        plt.xticks(range(len(topic_shares)), topic_names, rotation=45, ha='right')
        plt.title('Distribution of Topics Across Papers')
        plt.xlabel('Topics')
        plt.ylabel('Average Topic Share')
        
        if save:
            plt.savefig(self.output_dir / 'topic_distribution.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved topic distribution plot to {self.output_dir}")
        
        plt.close()
    
    def plot_trends(self, trends: pd.DataFrame, save: bool = True) -> None:
        """
        Plot research trends over time.
        
        Args:
            trends: DataFrame with trend data
            save: Whether to save the plot
        """
        plt.figure(figsize=(15, 8))
        
        # Plot trends
        for column in trends.columns:
            plt.plot(trends.index, trends[column], label=column, marker='o')
        
        plt.title('Research Trends by Category')
        plt.xlabel('Date')
        plt.ylabel('Number of Papers')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(self.output_dir / 'research_trends.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved research trends plot to {self.output_dir}")
        
        plt.close()
    
    def plot_connections(self, connections: Dict, save: bool = True) -> None:
        """
        Plot cross-category connections.
        
        Args:
            connections: Dictionary of category connections
            save: Whether to save the plot
        """
        # Create matrix from connections
        categories = sorted(set(cat for pair in connections.keys() for cat in pair))
        n_cats = len(categories)
        matrix = np.zeros((n_cats, n_cats))
        
        for (cat1, cat2), count in connections.items():
            i = categories.index(cat1)
            j = categories.index(cat2)
            matrix[i, j] = count
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(matrix,
                   xticklabels=categories,
                   yticklabels=categories,
                   annot=True,
                   fmt='g',
                   cmap='YlOrRd')
        
        plt.title('Cross-Category Connections')
        plt.xlabel('Target Category')
        plt.ylabel('Source Category')
        
        if save:
            plt.savefig(self.output_dir / 'category_connections.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved category connections plot to {self.output_dir}")
        
        plt.close()
    
    def create_dashboard(self, report: Dict, save: bool = True) -> None:
        """
        Create a comprehensive dashboard of visualizations.
        
        Args:
            report: Analysis report dictionary
            save: Whether to save the dashboard
        """
        plt.figure(figsize=(20, 15))
        
        # Create subplots
        gs = plt.GridSpec(3, 2)
        
        # Category distribution
        plt.subplot(gs[0, 0])
        categories = pd.Series(report['categories'])
        sns.barplot(x=categories.values, y=categories.index)
        plt.title('Papers by Category')
        
        # Network statistics
        plt.subplot(gs[0, 1])
        stats = pd.Series(report['network_stats'])
        sns.barplot(x=stats.values, y=stats.index)
        plt.title('Network Statistics')
        
        # Topic distribution
        plt.subplot(gs[1, :])
        topics = report['topics']
        topic_names = [f"Topic {i+1}" for i in range(len(topics))]
        plt.text(0.1, 0.9, '\n'.join([f"{name}: {', '.join(words[:5])}"
                                     for name, words in zip(topic_names, topics)]),
                fontsize=10, transform=plt.gca().transAxes)
        plt.axis('off')
        plt.title('Topic Descriptions')
        
        # Trends
        plt.subplot(gs[2, :])
        trends = pd.DataFrame(report['trends'])
        for column in trends.columns:
            plt.plot(trends.index, trends[column], label=column)
        plt.title('Research Trends')
        plt.legend(bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / 'dashboard.png', dpi=300, bbox_inches='tight')
            logger.info(f"Saved dashboard to {self.output_dir}")
        
        plt.close()
