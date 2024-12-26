import pandas as pd
import numpy as np
from typing import Dict, List, Any
import os
from datetime import datetime
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class ReportGenerator:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.latest_data_dir = self._get_latest_data_dir()
        self.report_sections = []
        
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
                source = file.replace('_scraper_data.csv', '')
                data[source] = pd.read_csv(path)
        return data

    def analyze_research_trends(self) -> str:
        """Analyze research trends and patterns"""
        data = self._load_data()
        
        # Combine all descriptions
        all_descriptions = []
        for df in data.values():
            if 'description' in df.columns:
                all_descriptions.extend(df['description'].dropna().tolist())
        
        # Extract key topics
        tfidf = TfidfVectorizer(max_features=20, stop_words='english')
        tfidf_matrix = tfidf.fit_transform(all_descriptions)
        
        # Get top terms
        feature_names = tfidf.get_feature_names_out()
        scores = np.mean(tfidf_matrix.toarray(), axis=0)
        top_terms = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
        
        # Format findings
        trends_report = "## Research Trends Analysis\n\n"
        trends_report += "### Key Research Topics\n"
        trends_report += "Based on TF-IDF analysis of research descriptions, the following topics are most prominent:\n\n"
        
        for term, score in top_terms[:10]:
            trends_report += f"- {term.capitalize()}: {score:.4f} relevance score\n"
            
        return trends_report

    def analyze_technology_relationships(self) -> str:
        """Analyze relationships between different technologies"""
        data = self._load_data()
        
        # Define key technology areas
        tech_areas = [
            'quantum', 'propulsion', 'field', 'energy', 'material',
            'containment', 'antimatter', 'spacetime', 'warp', 'shield'
        ]
        
        # Calculate co-occurrence matrix
        cooccurrence = np.zeros((len(tech_areas), len(tech_areas)))
        
        for df in data.values():
            if 'description' in df.columns:
                for desc in df['description'].dropna():
                    for i, tech1 in enumerate(tech_areas):
                        for j, tech2 in enumerate(tech_areas):
                            if tech1 in desc.lower() and tech2 in desc.lower():
                                cooccurrence[i,j] += 1
        
        # Format findings
        tech_report = "## Technology Relationship Analysis\n\n"
        tech_report += "### Key Technology Pairs\n"
        tech_report += "The following technology combinations show strong relationships:\n\n"
        
        # Get top pairs
        pairs = []
        for i in range(len(tech_areas)):
            for j in range(i+1, len(tech_areas)):
                if cooccurrence[i,j] > 0:
                    pairs.append((tech_areas[i], tech_areas[j], cooccurrence[i,j]))
        
        pairs.sort(key=lambda x: x[2], reverse=True)
        
        for tech1, tech2, count in pairs[:10]:
            tech_report += f"- {tech1.capitalize()} + {tech2.capitalize()}: {int(count)} co-occurrences\n"
            
        return tech_report

    def analyze_research_timeline(self) -> str:
        """Analyze research timeline and development patterns"""
        data = self._load_data()
        
        timeline_report = "## Research Timeline Analysis\n\n"
        timeline_report += "### Research Development Patterns\n"
        
        # Combine all data with dates
        timeline_data = []
        for source, df in data.items():
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                timeline_data.append(df[['date', 'title', 'source']])
        
        if timeline_data:
            combined_data = pd.concat(timeline_data)
            combined_data = combined_data.sort_values('date')
            
            # Analyze publication patterns
            timeline_report += "\n#### Publication Patterns\n"
            for source in combined_data['source'].unique():
                source_data = combined_data[combined_data['source'] == source]
                timeline_report += f"- {source}: {len(source_data)} publications\n"
            
            # Analyze time periods
            timeline_report += "\n#### Key Time Periods\n"
            year_counts = combined_data['date'].dt.year.value_counts().sort_index()
            for year, count in year_counts.items():
                timeline_report += f"- {year}: {count} publications\n"
            
        return timeline_report

    def analyze_source_contributions(self) -> str:
        """Analyze contributions from different sources"""
        data = self._load_data()
        
        source_report = "## Source Analysis\n\n"
        source_report += "### Contributions by Source\n"
        
        # Calculate contributions
        source_stats = {}
        for source, df in data.items():
            stats = {
                'total_entries': len(df),
                'unique_topics': len(df['title'].unique()) if 'title' in df.columns else 0,
            }
            source_stats[source] = stats
        
        # Format findings
        for source, stats in source_stats.items():
            source_report += f"\n#### {source}\n"
            source_report += f"- Total Entries: {stats['total_entries']}\n"
            source_report += f"- Unique Topics: {stats['unique_topics']}\n"
            
        return source_report

    def generate_report(self) -> str:
        """Generate complete analysis report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Warp Drive Research Analysis Report
Generated: {timestamp}

## Executive Summary
This report analyzes the relationships and patterns in warp drive research data collected from various sources.

"""
        # Add analysis sections
        report += self.analyze_research_trends()
        report += "\n\n" + self.analyze_technology_relationships()
        report += "\n\n" + self.analyze_research_timeline()
        report += "\n\n" + self.analyze_source_contributions()
        
        # Add recommendations section
        report += """
## Recommendations

### Research Focus Areas
1. Investigate strongly correlated technology pairs
2. Focus on emerging research trends
3. Fill gaps in underrepresented areas

### Technology Development
1. Prioritize development of key enabling technologies
2. Focus on integration of related technology areas
3. Address technical challenges in high-impact areas

### Future Directions
1. Monitor emerging research trends
2. Expand data collection to new sources
3. Update analysis as new research emerges
"""
        
        # Save report
        report_path = os.path.join(self.output_dir, 'research_analysis_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        return report_path
