import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import os

class DataVisualizer:
    def __init__(self, data_dir: str, output_dir: str):
        """Initialize the visualizer with data and output directories"""
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.latest_data_dir = self._get_latest_data_dir()
        
    def _get_latest_data_dir(self) -> str:
        """Get the most recent data directory"""
        data_dirs = [d for d in os.listdir(self.data_dir) 
                    if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith('scrape_')]
        if not data_dirs:
            raise ValueError("No data directories found")
        return os.path.join(self.data_dir, sorted(data_dirs)[-1])
        
    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """Load data from CSV files"""
        data = {}
        for file in os.listdir(self.latest_data_dir):
            if file.endswith('.csv'):
                path = os.path.join(self.latest_data_dir, file)
                source = file.replace('scraper_data.csv', '')
                data[source] = pd.read_csv(path)
        return data
        
    def create_concept_network(self):
        """Create a network visualization of related concepts"""
        data = self._load_data()
        
        # Combine all descriptions
        all_text = []
        for df in data.values():
            if 'description' in df.columns:
                all_text.extend(df['description'].dropna().tolist())
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(all_text)
        
        # Create network
        G = nx.Graph()
        feature_names = tfidf.get_feature_names_out()
        
        # Add nodes
        for word in feature_names:
            G.add_node(word)
        
        # Add edges based on co-occurrence
        for i, word1 in enumerate(feature_names):
            for j, word2 in enumerate(feature_names[i+1:], i+1):
                weight = np.dot(tfidf_matrix[:,i].toarray().flatten(),
                              tfidf_matrix[:,j].toarray().flatten())
                if weight > 0.1:  # Threshold for connection
                    G.add_edge(word1, word2, weight=weight)
        
        # Create visualization
        plt.figure(figsize=(20, 20))
        pos = nx.spring_layout(G)
        
        # Draw network
        nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='lightblue')
        nx.draw_networkx_edges(G, pos, width=[G[u][v]['weight']*5 for u,v in G.edges()])
        nx.draw_networkx_labels(G, pos)
        
        plt.title("Concept Relationship Network")
        plt.savefig(os.path.join(self.output_dir, 'concept_network.png'))
        plt.close()
        
    def create_technology_heatmap(self):
        """Create a heatmap of technology relationships"""
        data = self._load_data()
        
        # Define key technology areas
        tech_areas = [
            'quantum', 'propulsion', 'field', 'energy', 'material',
            'containment', 'antimatter', 'spacetime', 'warp', 'shield'
        ]
        
        # Create correlation matrix
        correlation_matrix = np.zeros((len(tech_areas), len(tech_areas)))
        
        # Calculate co-occurrence
        for df in data.values():
            if 'description' in df.columns:
                text = ' '.join(df['description'].dropna().astype(str))
                for i, tech1 in enumerate(tech_areas):
                    for j, tech2 in enumerate(tech_areas):
                        correlation_matrix[i,j] += text.count(f"{tech1}.*{tech2}")
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, xticklabels=tech_areas, yticklabels=tech_areas,
                   cmap='YlOrRd', annot=True)
        
        plt.title("Technology Relationship Heatmap")
        plt.savefig(os.path.join(self.output_dir, 'technology_heatmap.png'))
        plt.close()
        
    def create_research_timeline(self):
        """Create a timeline of research developments"""
        data = self._load_data()
        
        # Combine all data with dates
        timeline_data = []
        for source, df in data.items():
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                timeline_data.append(df[['date', 'title', 'source']])
        
        if timeline_data:
            combined_data = pd.concat(timeline_data)
            combined_data = combined_data.sort_values('date')
            
            # Create timeline visualization
            fig = go.Figure()
            
            for source in combined_data['source'].unique():
                source_data = combined_data[combined_data['source'] == source]
                fig.add_trace(go.Scatter(
                    x=source_data['date'],
                    y=[source] * len(source_data),
                    mode='markers',
                    name=source,
                    text=source_data['title'],
                    hoverinfo='text'
                ))
            
            fig.update_layout(
                title="Research Development Timeline",
                xaxis_title="Date",
                yaxis_title="Source",
                height=800
            )
            
            fig.write_html(os.path.join(self.output_dir, 'research_timeline.html'))
        
    def create_topic_clusters(self):
        """Create topic clusters visualization"""
        data = self._load_data()
        
        # Combine all descriptions
        all_text = []
        for df in data.values():
            if 'description' in df.columns:
                all_text.extend(df['description'].dropna().tolist())
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(all_text)
        
        # Perform PCA
        pca = PCA(n_components=2)
        coords = pca.fit_transform(tfidf_matrix.toarray())
        
        # Create scatter plot
        plt.figure(figsize=(15, 10))
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.5)
        
        # Add labels for some points
        for i, word in enumerate(tfidf.get_feature_names_out()):
            if i % 5 == 0:  # Label every 5th word to avoid overcrowding
                plt.annotate(word, (coords[i, 0], coords[i, 1]))
        
        plt.title("Research Topic Clusters")
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.savefig(os.path.join(self.output_dir, 'topic_clusters.png'))
        plt.close()
        
    def create_wordcloud(self):
        """Create word cloud of key concepts"""
        data = self._load_data()
        
        # Combine all text
        text = ""
        for df in data.values():
            if 'description' in df.columns:
                text += " ".join(df['description'].dropna().astype(str))
        
        # Create word cloud
        wordcloud = WordCloud(width=1600, height=800, background_color='white').generate(text)
        
        plt.figure(figsize=(20,10))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title("Key Concepts Word Cloud")
        plt.savefig(os.path.join(self.output_dir, 'wordcloud.png'))
        plt.close()
        
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        self.create_concept_network()
        self.create_technology_heatmap()
        self.create_research_timeline()
        self.create_topic_clusters()
        self.create_wordcloud()
