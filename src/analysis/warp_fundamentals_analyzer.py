import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import os
from datetime import datetime
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx

class WarpFundamentalsAnalyzer:
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.latest_data_dir = self._get_latest_data_dir()
        
        # Core warp drive components
        self.core_components = {
            'spacetime_manipulation': [
                'alcubierre metric',
                'spacetime curvature',
                'metric tensor',
                'warp bubble',
                'spacetime geometry',
                'lorentz transformation',
                'gravitational field'
            ],
            'energy_systems': [
                'negative energy',
                'zero point energy',
                'quantum vacuum',
                'energy density',
                'power generation',
                'antimatter reaction',
                'energy conversion'
            ],
            'field_generation': [
                'warp field',
                'containment field',
                'gravitational field',
                'electromagnetic field',
                'field generator',
                'field stability',
                'field geometry'
            ],
            'exotic_matter': [
                'negative mass',
                'exotic matter',
                'dark energy',
                'negative energy density',
                'quantum fields',
                'casimir effect',
                'vacuum energy'
            ],
            'propulsion_systems': [
                'warp propulsion',
                'antimatter drive',
                'field propulsion',
                'quantum propulsion',
                'space drive',
                'propulsion efficiency',
                'thrust generation'
            ]
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

    def analyze_core_components(self) -> str:
        """Analyze research coverage of core warp drive components"""
        data = self._load_data()
        
        # Initialize component coverage tracking
        coverage = defaultdict(lambda: defaultdict(int))
        
        # Analyze each component's coverage in the research
        for component, keywords in self.core_components.items():
            for df in data.values():
                if 'description' in df.columns:
                    for keyword in keywords:
                        mask = df['description'].str.contains(keyword, case=False, na=False)
                        coverage[component][keyword] = mask.sum()
        
        # Format findings
        report = "## Core Warp Drive Components Analysis\n\n"
        
        for component, keyword_counts in coverage.items():
            total_mentions = sum(keyword_counts.values())
            report += f"\n### {component.replace('_', ' ').title()}\n"
            report += f"Total Research Mentions: {total_mentions}\n\n"
            report += "Key Concepts Coverage:\n"
            
            # Sort keywords by mention count
            sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            for keyword, count in sorted_keywords:
                report += f"- {keyword}: {count} mentions\n"
                
        return report

    def analyze_theoretical_foundations(self) -> str:
        """Analyze theoretical physics foundations"""
        data = self._load_data()
        
        # Key theoretical concepts
        theoretical_concepts = {
            'general_relativity': [
                'einstein field equations',
                'spacetime curvature',
                'metric tensor',
                'gravitational field',
                'lorentz transformation'
            ],
            'quantum_mechanics': [
                'quantum field theory',
                'quantum vacuum',
                'quantum fluctuations',
                'quantum effects',
                'quantum tunneling'
            ],
            'field_theory': [
                'field equations',
                'field dynamics',
                'field interaction',
                'field strength',
                'field geometry'
            ]
        }
        
        # Analyze theoretical foundations
        theory_coverage = defaultdict(lambda: defaultdict(int))
        
        for theory, concepts in theoretical_concepts.items():
            for df in data.values():
                if 'description' in df.columns:
                    for concept in concepts:
                        mask = df['description'].str.contains(concept, case=False, na=False)
                        theory_coverage[theory][concept] = mask.sum()
        
        # Format findings
        report = "## Theoretical Foundations Analysis\n\n"
        
        for theory, concept_counts in theory_coverage.items():
            report += f"\n### {theory.replace('_', ' ').title()}\n"
            total_mentions = sum(concept_counts.values())
            report += f"Total Research Coverage: {total_mentions} mentions\n\n"
            
            # Sort concepts by mention count
            sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
            for concept, count in sorted_concepts:
                report += f"- {concept}: {count} mentions\n"
                
        return report

    def analyze_technical_challenges(self) -> str:
        """Analyze technical challenges and potential solutions"""
        data = self._load_data()
        
        # Define key challenges
        technical_challenges = {
            'energy_requirements': [
                'energy density',
                'power generation',
                'energy efficiency',
                'power requirements',
                'energy source'
            ],
            'field_stability': [
                'field stability',
                'containment',
                'field geometry',
                'field control',
                'field maintenance'
            ],
            'exotic_matter_creation': [
                'negative mass',
                'exotic matter production',
                'negative energy',
                'matter synthesis',
                'quantum effects'
            ],
            'spacetime_manipulation': [
                'metric engineering',
                'spacetime distortion',
                'bubble formation',
                'curvature control',
                'geometric stability'
            ]
        }
        
        # Analyze challenges and solutions
        challenge_analysis = defaultdict(lambda: defaultdict(list))
        
        for challenge, keywords in technical_challenges.items():
            for df in data.values():
                if 'description' in df.columns and 'title' in df.columns:
                    for keyword in keywords:
                        mask = df['description'].str.contains(keyword, case=False, na=False)
                        if mask.any():
                            relevant_papers = df[mask][['title', 'description']]
                            challenge_analysis[challenge][keyword].extend(
                                relevant_papers.to_dict('records')
                            )
        
        # Format findings
        report = "## Technical Challenges Analysis\n\n"
        
        for challenge, keyword_papers in challenge_analysis.items():
            report += f"\n### {challenge.replace('_', ' ').title()}\n"
            total_papers = sum(len(papers) for papers in keyword_papers.values())
            report += f"Total Related Research: {total_papers} papers\n\n"
            
            # Key findings for each challenge
            report += "Key Research Areas:\n"
            for keyword, papers in keyword_papers.items():
                if papers:
                    report += f"\n#### {keyword.replace('_', ' ').title()}\n"
                    report += f"Number of related papers: {len(papers)}\n"
                    # Include a few key paper titles
                    for paper in papers[:3]:
                        report += f"- {paper['title']}\n"
                        
        return report

    def analyze_development_roadmap(self) -> str:
        """Analyze research progression and development roadmap"""
        data = self._load_data()
        
        # Development stages
        stages = {
            'theoretical_foundation': [
                'theoretical framework',
                'mathematical model',
                'physics theory',
                'theoretical analysis'
            ],
            'concept_validation': [
                'experimental validation',
                'proof of concept',
                'concept demonstration',
                'theoretical proof'
            ],
            'technology_development': [
                'technology development',
                'prototype',
                'engineering design',
                'technical implementation'
            ],
            'system_integration': [
                'system integration',
                'component integration',
                'system assembly',
                'integration testing'
            ],
            'testing_validation': [
                'testing',
                'validation',
                'performance analysis',
                'system verification'
            ]
        }
        
        # Analyze development stages
        stage_progress = defaultdict(lambda: defaultdict(list))
        
        for stage, keywords in stages.items():
            for df in data.values():
                if 'description' in df.columns and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    for keyword in keywords:
                        mask = df['description'].str.contains(keyword, case=False, na=False)
                        if mask.any():
                            relevant_research = df[mask][['title', 'date', 'description']]
                            stage_progress[stage][keyword].extend(
                                relevant_research.to_dict('records')
                            )
        
        # Format findings
        report = "## Development Roadmap Analysis\n\n"
        
        for stage, keyword_research in stage_progress.items():
            report += f"\n### {stage.replace('_', ' ').title()} Stage\n"
            total_research = sum(len(research) for research in keyword_research.values())
            report += f"Total Research Items: {total_research}\n\n"
            
            # Progress in each area
            report += "Research Progress:\n"
            for keyword, research in keyword_research.items():
                if research:
                    report += f"\n#### {keyword.replace('_', ' ').title()}\n"
                    report += f"Number of research items: {len(research)}\n"
                    # Show recent developments
                    sorted_research = sorted(research, key=lambda x: x['date'], reverse=True)
                    for item in sorted_research[:2]:
                        report += f"- {item['title']} ({item['date'][:10]})\n"
                        
        return report

    def generate_warp_fundamentals_report(self) -> str:
        """Generate comprehensive warp drive fundamentals report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# Warp Drive Fundamentals Analysis Report
Generated: {timestamp}

## Executive Summary
This report analyzes the fundamental components, theoretical foundations, and technical challenges
in warp drive development based on comprehensive research data analysis.

"""
        # Add analysis sections
        report += self.analyze_core_components()
        report += "\n\n" + self.analyze_theoretical_foundations()
        report += "\n\n" + self.analyze_technical_challenges()
        report += "\n\n" + self.analyze_development_roadmap()
        
        # Add conclusions and next steps
        report += """
## Key Findings and Implications

### Critical Technologies
1. Negative energy generation and control
2. Field generation and stability systems
3. Spacetime metric engineering
4. Exotic matter production and containment

### Research Priorities
1. Theoretical framework validation
2. Energy requirement optimization
3. Field stability enhancement
4. Material science advancement

### Development Path
1. Continue theoretical physics research
2. Develop small-scale experiments
3. Advance enabling technologies
4. Focus on system integration

## Next Steps

### Immediate Focus
1. Enhance negative energy research
2. Develop field generation prototypes
3. Improve exotic matter production
4. Advance containment technologies

### Long-term Goals
1. Validate theoretical models
2. Demonstrate key technologies
3. Integrate core systems
4. Achieve stable warp fields
"""
        
        # Save report
        report_path = os.path.join(self.output_dir, 'warp_fundamentals_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        return report_path
