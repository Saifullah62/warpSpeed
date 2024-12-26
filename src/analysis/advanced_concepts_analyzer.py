import pandas as pd
import numpy as np
from typing import Dict, List, Any
import os
from datetime import datetime
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class AdvancedConceptsAnalyzer:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.latest_data_dir = self._get_latest_data_dir()
        
        # Research focus areas
        self.focus_areas = {
            'zero_point_energy': {
                'key_concepts': [
                    'quantum vacuum',
                    'zero-point field',
                    'vacuum fluctuations',
                    'energy extraction',
                    'casimir effect'
                ],
                'metrics': [
                    'energy density',
                    'extraction efficiency',
                    'stability duration',
                    'power output',
                    'system scale'
                ]
            },
            'spacetime_manipulation': {
                'key_concepts': [
                    'metric engineering',
                    'field geometry',
                    'curvature control',
                    'non-exotic solutions',
                    'warp bubble'
                ],
                'metrics': [
                    'curvature strength',
                    'field stability',
                    'energy requirements',
                    'geometric precision',
                    'bubble integrity'
                ]
            },
            'integrated_systems': {
                'key_concepts': [
                    'field harmonization',
                    'navigation control',
                    'shield integration',
                    'bubble properties',
                    'system synergy'
                ],
                'metrics': [
                    'navigation accuracy',
                    'shield effectiveness',
                    'system efficiency',
                    'response time',
                    'field coherence'
                ]
            }
        }
        
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

    def analyze_research_progress(self) -> str:
        """Analyze research progress in each focus area"""
        data = self._load_data()
        
        progress_report = "## Research Progress Analysis\n\n"
        
        for area, concepts in self.focus_areas.items():
            progress_report += f"\n### {area.replace('_', ' ').title()}\n\n"
            
            # Analyze concept coverage
            concept_coverage = defaultdict(int)
            for df in data.values():
                if 'summary' in df.columns:
                    for concept in concepts['key_concepts']:
                        mask = df['summary'].str.contains(concept, case=False, na=False)
                        concept_coverage[concept] += mask.sum()
            
            progress_report += "#### Key Concept Coverage\n"
            for concept, count in sorted(concept_coverage.items(), key=lambda x: x[1], reverse=True):
                progress_report += f"- {concept}: {count} research items\n"
            
            # Analyze metrics discussion
            metric_coverage = defaultdict(int)
            for df in data.values():
                if 'summary' in df.columns:
                    for metric in concepts['metrics']:
                        mask = df['summary'].str.contains(metric, case=False, na=False)
                        metric_coverage[metric] += mask.sum()
            
            progress_report += "\n#### Performance Metrics Coverage\n"
            for metric, count in sorted(metric_coverage.items(), key=lambda x: x[1], reverse=True):
                progress_report += f"- {metric}: {count} research items\n"
            
        return progress_report

    def analyze_technical_breakthroughs(self) -> str:
        """Analyze significant technical breakthroughs"""
        data = self._load_data()
        
        breakthroughs_report = "## Technical Breakthroughs Analysis\n\n"
        
        # Keywords indicating breakthroughs
        breakthrough_indicators = [
            'breakthrough', 'discovery', 'novel', 'innovative',
            'new method', 'improvement', 'advancement', 'progress'
        ]
        
        for area, concepts in self.focus_areas.items():
            breakthroughs_report += f"\n### {area.replace('_', ' ').title()}\n\n"
            
            area_breakthroughs = []
            for df in data.values():
                if 'summary' in df.columns and 'title' in df.columns:
                    # Find papers with breakthrough indicators
                    for indicator in breakthrough_indicators:
                        mask = (df['summary'].str.contains(indicator, case=False, na=False) |
                               df['title'].str.contains(indicator, case=False, na=False))
                        if mask.any():
                            relevant_papers = df[mask][['title', 'summary', 'published']]
                            area_breakthroughs.extend(relevant_papers.to_dict('records'))
            
            # Sort by date and show most recent
            if area_breakthroughs:
                sorted_breakthroughs = sorted(area_breakthroughs, 
                                            key=lambda x: x['published'],
                                            reverse=True)
                
                breakthroughs_report += "#### Recent Breakthroughs\n"
                for breakthrough in sorted_breakthroughs[:5]:
                    breakthroughs_report += f"- {breakthrough['title']} ({breakthrough['published'][:10]})\n"
                    
        return breakthroughs_report

    def analyze_integration_opportunities(self) -> str:
        """Analyze opportunities for system integration"""
        data = self._load_data()
        
        integration_report = "## Integration Opportunities Analysis\n\n"
        
        # Create concept similarity network
        all_concepts = []
        for concepts in self.focus_areas.values():
            all_concepts.extend(concepts['key_concepts'])
        
        # Create concept co-occurrence matrix
        concept_matrix = np.zeros((len(all_concepts), len(all_concepts)))
        
        for df in data.values():
            if 'summary' in df.columns:
                for i, concept1 in enumerate(all_concepts):
                    for j, concept2 in enumerate(all_concepts):
                        if i != j:
                            co_occurrence = df['summary'].str.contains(concept1, case=False, na=False) & \
                                          df['summary'].str.contains(concept2, case=False, na=False)
                            concept_matrix[i,j] += co_occurrence.sum()
        
        # Find strong connections
        integration_report += "### Key Integration Points\n\n"
        for i, concept1 in enumerate(all_concepts):
            for j, concept2 in enumerate(all_concepts[i+1:], i+1):
                if concept_matrix[i,j] > 0:
                    integration_report += f"- {concept1} + {concept2}: {int(concept_matrix[i,j])} co-occurrences\n"
        
        return integration_report

    def analyze_development_challenges(self) -> str:
        """Analyze key development challenges"""
        data = self._load_data()
        
        challenges_report = "## Development Challenges Analysis\n\n"
        
        # Challenge indicators
        challenge_indicators = [
            'challenge', 'problem', 'difficulty', 'barrier',
            'limitation', 'constraint', 'obstacle', 'issue'
        ]
        
        for area, concepts in self.focus_areas.items():
            challenges_report += f"\n### {area.replace('_', ' ').title()} Challenges\n\n"
            
            area_challenges = []
            for df in data.values():
                if 'summary' in df.columns:
                    for indicator in challenge_indicators:
                        mask = df['summary'].str.contains(indicator, case=False, na=False)
                        if mask.any():
                            challenges = df[mask][['title', 'summary']]
                            area_challenges.extend(challenges.to_dict('records'))
            
            if area_challenges:
                challenges_report += "#### Key Challenges Identified\n"
                for challenge in area_challenges[:5]:
                    challenges_report += f"- {challenge['title']}\n"
                    
        return challenges_report

    def generate_advanced_concepts_report(self) -> str:
        """Generate comprehensive analysis report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Advanced Warp Drive Concepts Analysis
Generated: {timestamp}

## Executive Summary
This report analyzes advanced research in zero-point energy, alternative spacetime manipulation,
and integrated navigation/shielding systems for warp drive development.

"""
        # Add analysis sections
        report += self.analyze_research_progress()
        report += "\n\n" + self.analyze_technical_breakthroughs()
        report += "\n\n" + self.analyze_integration_opportunities()
        report += "\n\n" + self.analyze_development_challenges()
        
        # Add recommendations
        report += """
## Research Recommendations

### Zero-Point Energy Development
1. Focus on quantum vacuum energy extraction methods
2. Develop stable containment systems
3. Scale up energy harvesting capabilities
4. Optimize conversion efficiency

### Alternative Spacetime Manipulation
1. Explore non-exotic matter solutions
2. Develop metric engineering techniques
3. Improve field stability methods
4. Reduce energy requirements

### Integrated Systems
1. Harmonize navigation and shielding fields
2. Develop unified control systems
3. Optimize field configurations
4. Enhance system efficiency

## Next Steps

### Immediate Priorities
1. Prototype quantum vacuum energy extractors
2. Test alternative spacetime manipulation methods
3. Develop integrated system simulations

### Long-term Goals
1. Achieve stable zero-point energy extraction
2. Demonstrate practical spacetime manipulation
3. Create unified warp field system
"""
        
        # Save report
        report_path = os.path.join(self.output_dir, 'advanced_concepts_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        return report_path
