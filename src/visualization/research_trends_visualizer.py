import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Any
import os
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class ResearchTrendsVisualizer:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(self.viz_dir, exist_ok=True)
        
    def _load_data(self) -> pd.DataFrame:
        """Load and combine all research data"""
        dfs = []
        for file in os.listdir(self.data_dir):
            if file.endswith('.csv'):
                path = os.path.join(self.data_dir, file)
                df = pd.read_csv(path)
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def create_research_timeline(self) -> str:
        """Create an interactive timeline of research publications"""
        df = self._load_data()
        
        # Convert published dates to datetime
        df['published'] = pd.to_datetime(df['published'])
        df['year'] = df['published'].dt.year
        
        # Count publications by year and research area
        timeline_data = df.groupby(['year', 'research_area']).size().reset_index(name='count')
        
        # Create interactive line plot
        fig = px.line(timeline_data, 
                     x='year', 
                     y='count',
                     color='research_area',
                     title='Research Publications Timeline',
                     labels={'count': 'Number of Publications',
                            'year': 'Year',
                            'research_area': 'Research Area'},
                     template='plotly_dark')
        
        fig.update_layout(
            showlegend=True,
            legend_title_text='Research Area',
            xaxis_title='Year',
            yaxis_title='Number of Publications'
        )
        
        output_path = os.path.join(self.viz_dir, 'research_timeline.html')
        fig.write_html(output_path)
        return output_path

    def create_concept_network(self) -> str:
        """Create a network visualization of related concepts"""
        df = self._load_data()
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes for each research area
        research_areas = df['research_area'].unique()
        for area in research_areas:
            G.add_node(area, node_type='area', size=2000)
        
        # Add nodes for common keywords and connect them
        keywords = {
            'zero_point_energy': ['quantum vacuum', 'zero-point field', 'vacuum fluctuations', 
                                'casimir effect', 'energy extraction'],
            'spacetime_manipulation': ['metric engineering', 'warp field', 'spacetime curvature', 
                                     'field geometry', 'bubble dynamics'],
            'integrated_systems': ['navigation control', 'shield harmonization', 'field integration', 
                                 'system efficiency', 'bubble properties']
        }
        
        for area, area_keywords in keywords.items():
            for keyword in area_keywords:
                G.add_node(keyword, node_type='keyword', size=1000)
                G.add_edge(area, keyword, weight=1)
        
        # Calculate node positions using spring layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create plot
        plt.figure(figsize=(15, 15))
        plt.style.use('dark_background')
        
        # Draw nodes
        node_sizes = [G.nodes[node]['size'] for node in G.nodes()]
        node_colors = ['#1f77b4' if G.nodes[node]['node_type'] == 'area' else '#ff7f0e' 
                      for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.7)
        nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='white')
        nx.draw_networkx_labels(G, pos, font_size=10, font_color='white')
        
        plt.title('Warp Drive Research Concept Network', fontsize=16, color='white', pad=20)
        plt.axis('off')
        
        output_path = os.path.join(self.viz_dir, 'concept_network.png')
        plt.savefig(output_path, bbox_inches='tight', facecolor='black')
        plt.close()
        return output_path

    def create_breakthrough_heatmap(self) -> str:
        """Create a heatmap of breakthrough intensity across research areas"""
        df = self._load_data()
        
        # Define breakthrough indicators
        breakthrough_keywords = [
            'breakthrough', 'discovery', 'novel', 'innovative',
            'new method', 'improvement', 'advancement', 'progress'
        ]
        
        # Create matrix of breakthrough mentions
        breakthrough_data = defaultdict(lambda: defaultdict(int))
        
        for _, row in df.iterrows():
            area = row['research_area']
            summary = str(row['summary']).lower()
            
            for keyword in breakthrough_keywords:
                if keyword in summary:
                    breakthrough_data[area][keyword] += 1
        
        # Convert to DataFrame
        heatmap_df = pd.DataFrame(breakthrough_data).fillna(0)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        plt.style.use('dark_background')
        
        sns.heatmap(heatmap_df, 
                   cmap='YlOrRd',
                   annot=True,
                   fmt='g',
                   cbar_kws={'label': 'Number of Mentions'})
        
        plt.title('Research Breakthrough Intensity Heatmap', pad=20)
        plt.xlabel('Research Area')
        plt.ylabel('Breakthrough Indicator')
        
        output_path = os.path.join(self.viz_dir, 'breakthrough_heatmap.png')
        plt.savefig(output_path, bbox_inches='tight', facecolor='black')
        plt.close()
        return output_path

    def create_technology_readiness(self) -> str:
        """Create a radar chart of technology readiness levels"""
        # Technology readiness assessment (hypothetical scores)
        categories = [
            'Theoretical Foundation',
            'Experimental Validation',
            'Component Development',
            'System Integration',
            'Performance Testing',
            'Scalability'
        ]
        
        research_areas = {
            'Zero-Point Energy': [9, 5, 4, 3, 2, 2],
            'Spacetime Manipulation': [8, 4, 3, 2, 2, 1],
            'Integrated Systems': [7, 4, 3, 3, 2, 2]
        }
        
        # Create radar chart
        fig = go.Figure()
        
        for area, scores in research_areas.items():
            fig.add_trace(go.Scatterpolar(
                r=scores + [scores[0]],  # Complete the circle
                theta=categories + [categories[0]],  # Complete the circle
                name=area,
                fill='toself'
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title='Technology Readiness Assessment',
            template='plotly_dark'
        )
        
        output_path = os.path.join(self.viz_dir, 'technology_readiness.html')
        fig.write_html(output_path)
        return output_path

    def create_research_focus_treemap(self) -> str:
        """Create a treemap of research focus areas and their components"""
        research_hierarchy = {
            'Zero-Point Energy': {
                'Quantum Vacuum': 35,
                'Field Fluctuations': 25,
                'Energy Extraction': 20,
                'Containment Systems': 15,
                'Scaling Methods': 5
            },
            'Spacetime Manipulation': {
                'Metric Engineering': 30,
                'Field Geometry': 25,
                'Non-Exotic Solutions': 20,
                'Bubble Dynamics': 15,
                'Stability Control': 10
            },
            'Integrated Systems': {
                'Navigation Control': 25,
                'Shield Harmonization': 25,
                'Field Integration': 20,
                'System Efficiency': 15,
                'Performance Optimization': 15
            }
        }
        
        # Prepare data for treemap
        labels = []
        parents = []
        values = []
        
        for area, components in research_hierarchy.items():
            labels.append(area)
            parents.append('')
            values.append(sum(components.values()))
            
            for component, value in components.items():
                labels.append(component)
                parents.append(area)
                values.append(value)
        
        fig = go.Figure(go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            textinfo="label+value",
            marker=dict(
                colors=['#1f77b4', '#ff7f0e', '#2ca02c',
                       '#d62728', '#9467bd', '#8c564b']
            )
        ))
        
        fig.update_layout(
            title='Research Focus Distribution',
            template='plotly_dark'
        )
        
        output_path = os.path.join(self.viz_dir, 'research_focus_treemap.html')
        fig.write_html(output_path)
        return output_path

    def generate_all_visualizations(self) -> List[str]:
        """Generate all visualizations and return paths"""
        print("Generating research visualizations...")
        
        paths = []
        
        print("1. Creating research timeline...")
        paths.append(self.create_research_timeline())
        
        print("2. Creating concept network...")
        paths.append(self.create_concept_network())
        
        print("3. Creating breakthrough heatmap...")
        paths.append(self.create_breakthrough_heatmap())
        
        print("4. Creating technology readiness assessment...")
        paths.append(self.create_technology_readiness())
        
        print("5. Creating research focus treemap...")
        paths.append(self.create_research_focus_treemap())
        
        return paths
