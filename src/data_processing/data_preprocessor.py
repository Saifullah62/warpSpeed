"""
Data preprocessing module for cleaning and integrating research paper data.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
import re
import numpy as np
from collections import defaultdict

@dataclass
class ResearchPaper:
    """Data class for standardized research paper representation."""
    title: str
    authors: List[str]
    abstract: str
    full_text: Optional[str]
    publication_date: str
    source: str  # arxiv, nasa, etc.
    document_id: str
    keywords: List[str]
    subject_categories: List[str]
    pdf_path: Optional[str]
    coherence_score: Optional[float] = None
    energy_metrics: Optional[Dict] = None
    geometric_params: Optional[Dict] = None

class DataPreprocessor:
    """Handles data cleaning, integration, and feature engineering for research papers."""
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.papers: List[ResearchPaper] = []
        
    def load_arxiv_papers(self) -> None:
        """Load and process arXiv papers."""
        arxiv_dir = self.data_dir / "arxiv"
        if not arxiv_dir.exists():
            print(f"Warning: arXiv directory not found at {arxiv_dir}")
            return

        # Load from pdfs directory which contains categorized papers
        pdfs_dir = arxiv_dir / "pdfs"
        if not pdfs_dir.exists():
            print(f"Warning: arXiv PDFs directory not found at {pdfs_dir}")
            return

        for category_dir in pdfs_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            category = category_dir.name
            print(f"Processing arXiv category: {category}")
            
            # Count files in category
            pdf_count = len(list(category_dir.glob("*.pdf")))
            print(f"Found {pdf_count} PDFs in {category}")
            
            # Create paper entries for each PDF
            for pdf_file in category_dir.glob("*.pdf"):
                # Extract paper ID from filename
                paper_id = pdf_file.stem
                
                self.papers.append(ResearchPaper(
                    title=self._extract_title_from_filename(pdf_file.name),
                    authors=[],  # Will be populated from metadata if available
                    abstract="",  # Will be populated from metadata if available
                    full_text=None,
                    publication_date=self._extract_date_from_filename(pdf_file.name),
                    source='arxiv',
                    document_id=paper_id,
                    keywords=self._extract_keywords_from_filename(pdf_file.name),
                    subject_categories=[category],
                    pdf_path=str(pdf_file)
                ))

    def load_nasa_papers(self) -> None:
        """Load and process NASA papers."""
        nasa_dir = self.data_dir / "nasa"
        if not nasa_dir.exists():
            print(f"Warning: NASA directory not found at {nasa_dir}")
            return

        # Load from pdfs directory which contains categorized papers
        pdfs_dir = nasa_dir / "pdfs"
        if not pdfs_dir.exists():
            print(f"Warning: NASA PDFs directory not found at {pdfs_dir}")
            return

        for category_dir in pdfs_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            category = category_dir.name
            print(f"Processing NASA category: {category}")
            
            # Process each text file and its corresponding metadata
            for txt_file in category_dir.glob("*.txt"):
                paper_id = txt_file.stem
                metadata_file = txt_file.parent / f"{paper_id}_metadata.json"
                
                # Read the text content
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        full_text = f.read().strip()
                except Exception as e:
                    print(f"Error reading text file {txt_file}: {e}")
                    continue

                # Read the metadata if available
                metadata = {}
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                    except Exception as e:
                        print(f"Error reading metadata file {metadata_file}: {e}")
                
                # Extract title from metadata or filename
                title = metadata.get('title', '') or self._extract_title_from_filename(txt_file.name)
                
                # Extract other metadata
                authors = metadata.get('authors', [])
                if isinstance(authors, str):
                    authors = [authors]
                
                abstract = metadata.get('abstract', '')
                publication_date = metadata.get('publication_date', '') or self._extract_date_from_filename(txt_file.name)
                keywords = metadata.get('keywords', [])
                if not keywords:
                    keywords = self._extract_keywords_from_text(title + " " + abstract + " " + full_text)
                
                self.papers.append(ResearchPaper(
                    title=self._clean_text(title),
                    authors=self._clean_authors(authors),
                    abstract=self._clean_text(abstract),
                    full_text=full_text,
                    publication_date=self._standardize_date(publication_date),
                    source='nasa',
                    document_id=paper_id,
                    keywords=keywords,
                    subject_categories=[category],
                    pdf_path=None  # NASA papers are stored as text
                ))
                
            print(f"Processed {len(list(category_dir.glob('*.txt')))} papers in {category}")

    def _extract_title_from_filename(self, filename: str) -> str:
        """Extract paper title from filename."""
        # Remove extension
        name = Path(filename).stem
        
        # Remove date pattern if present
        name = re.sub(r'\d{4}-\d{2}-\d{2}', '', name)
        
        # Replace underscores and hyphens with spaces
        name = name.replace('_', ' ').replace('-', ' ')
        
        # Clean up extra spaces
        name = ' '.join(name.split())
        
        return name

    def _extract_date_from_filename(self, filename: str) -> str:
        """Extract date from filename."""
        # Look for date pattern YYYY-MM-DD
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
        if date_match:
            return date_match.group(1)
            
        # Look for year pattern YYYY
        year_match = re.search(r'(\d{4})', filename)
        if year_match:
            return f"{year_match.group(1)}-01-01"
            
        return ""

    def _extract_keywords_from_filename(self, filename: str) -> List[str]:
        """Extract keywords from filename."""
        keywords = set()
        
        # Common keywords to look for
        important_terms = [
            'warp', 'drive', 'ftl', 'faster than light', 'quantum', 'spacetime',
            'relativity', 'field theory', 'propulsion', 'energy', 'metric',
            'topology', 'exotic matter', 'negative energy', 'antimatter'
        ]
        
        name = filename.lower()
        for term in important_terms:
            if term in name:
                keywords.add(term)
                
        return list(keywords)

    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract keywords from text content."""
        keywords = set()
        
        # Common keywords to look for
        important_terms = [
            'warp', 'drive', 'ftl', 'faster than light', 'quantum', 'spacetime',
            'relativity', 'field theory', 'propulsion', 'energy', 'metric',
            'topology', 'exotic matter', 'negative energy', 'antimatter',
            'spacecraft', 'interstellar', 'superluminal', 'alcubierre',
            'wormhole', 'field propulsion', 'space drive', 'breakthrough propulsion'
        ]
        
        text = text.lower()
        for term in important_terms:
            if term in text:
                keywords.add(term)
                
        return list(keywords)

    def _clean_text(self, text: str) -> str:
        """Clean and standardize text content."""
        if not text:
            return ""
        # Remove special characters and excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def _clean_authors(self, authors: List[str]) -> List[str]:
        """Clean and standardize author names."""
        cleaned = []
        for author in authors:
            if not author:
                continue
            # Remove special characters and standardize format
            author = re.sub(r'[^\w\s\-\.]', '', author)
            author = author.strip()
            if author:
                cleaned.append(author)
        return cleaned

    def _standardize_date(self, date_str: str) -> str:
        """Standardize date format."""
        if not date_str:
            return ""
        try:
            # Handle various date formats
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%Y']:
                try:
                    date_obj = datetime.strptime(date_str, fmt)
                    return date_obj.strftime('%Y-%m-%d')
                except ValueError:
                    continue
        except Exception:
            return ""
        return date_str

    def compute_features(self) -> None:
        """Compute additional features for each paper."""
        print("Computing features for papers...")
        for i, paper in enumerate(self.papers):
            if i % 100 == 0:
                print(f"Processed {i}/{len(self.papers)} papers")
                
            # Compute coherence score based on keyword relevance
            paper.coherence_score = self._compute_coherence(paper)
            
            # Extract energy-related metrics
            paper.energy_metrics = self._extract_energy_metrics(paper)
            
            # Extract geometric parameters
            paper.geometric_params = self._extract_geometric_params(paper)

    def _compute_coherence(self, paper: ResearchPaper) -> float:
        """Compute coherence score based on keyword relevance and content."""
        score = 0.0
        relevant_terms = {
            'warp drive': 5.0,
            'ftl': 4.0,
            'faster than light': 4.0,
            'quantum': 3.0,
            'spacetime': 4.0,
            'relativity': 3.0,
            'propulsion': 3.0,
            'energy': 2.0,
            'metric': 2.0,
            'exotic matter': 4.0,
            'negative energy': 3.0,
            'antimatter': 2.0
        }
        
        # Score based on keywords
        for keyword in paper.keywords:
            for term, weight in relevant_terms.items():
                if term in keyword.lower():
                    score += weight
                    
        # Score based on title
        for term, weight in relevant_terms.items():
            if term in paper.title.lower():
                score += weight * 1.5  # Title matches are more important
                    
        # Normalize score
        max_possible_score = sum(relevant_terms.values()) * 2.5  # Account for title bonus
        return min(score / max_possible_score, 1.0)

    def _extract_energy_metrics(self, paper: ResearchPaper) -> Dict:
        """Extract energy-related metrics from paper content."""
        metrics = {
            'negative_energy_density': None,
            'energy_requirements': None,
            'energy_efficiency': None,
            'has_energy_discussion': any(term in paper.title.lower() for term in ['energy', 'power', 'joule'])
        }
        return metrics

    def _extract_geometric_params(self, paper: ResearchPaper) -> Dict:
        """Extract geometric parameters from paper content."""
        params = {
            'spacetime_curvature': None,
            'metric_properties': None,
            'topology_class': None,
            'has_geometry_discussion': any(term in paper.title.lower() for term in ['metric', 'geometry', 'curvature', 'topology'])
        }
        return params

    def remove_duplicates(self) -> None:
        """Remove duplicate papers based on title and content similarity."""
        print("Removing duplicate papers...")
        seen_titles = {}  # title -> paper with highest coherence score
        
        for paper in self.papers:
            # Create a normalized version of the title for comparison
            norm_title = paper.title.lower().strip()
            
            if norm_title in seen_titles:
                # Keep the paper with the higher coherence score
                if paper.coherence_score > seen_titles[norm_title].coherence_score:
                    seen_titles[norm_title] = paper
            else:
                seen_titles[norm_title] = paper
        
        old_count = len(self.papers)
        self.papers = list(seen_titles.values())
        print(f"Removed {old_count - len(self.papers)} duplicate papers")

    def filter_irrelevant(self) -> None:
        """Filter out papers that are not relevant to warp drive research."""
        print("Filtering irrelevant papers...")
        min_coherence_score = 0.2  # Threshold for relevance
        
        old_count = len(self.papers)
        self.papers = [p for p in self.papers if p.coherence_score >= min_coherence_score]
        print(f"Filtered out {old_count - len(self.papers)} irrelevant papers")

    def to_huggingface_format(self) -> List[Dict]:
        """Convert papers to Hugging Face dataset format."""
        print("Converting to Hugging Face format...")
        dataset = []
        for paper in self.papers:
            entry = {
                'title': paper.title,
                'authors': paper.authors,
                'abstract': paper.abstract,
                'full_text': paper.full_text,
                'publication_date': paper.publication_date,
                'source': paper.source,
                'document_id': paper.document_id,
                'keywords': paper.keywords,
                'subject_categories': paper.subject_categories,
                'pdf_path': paper.pdf_path,
                'features': {
                    'coherence_score': paper.coherence_score,
                    'energy_metrics': paper.energy_metrics,
                    'geometric_params': paper.geometric_params
                }
            }
            dataset.append(entry)
        return dataset

    def save_processed_data(self, output_path: Union[str, Path]) -> None:
        """Save processed data to JSON file."""
        print(f"Saving processed data to {output_path}...")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        dataset = self.to_huggingface_format()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

    def generate_statistics(self) -> Dict:
        """Generate statistics about the processed dataset."""
        print("Generating dataset statistics...")
        stats = {
            'total_papers': len(self.papers),
            'papers_by_source': defaultdict(int),
            'papers_by_category': defaultdict(int),
            'date_range': {'earliest': None, 'latest': None},
            'avg_coherence_score': 0.0,
            'keyword_frequency': defaultdict(int),
            'categories_distribution': defaultdict(int),
            'energy_metrics_stats': {
                'papers_with_energy_discussion': 0
            },
            'geometric_params_stats': {
                'papers_with_geometry_discussion': 0
            }
        }
        
        coherence_scores = []
        
        for paper in self.papers:
            # Basic counts
            stats['papers_by_source'][paper.source] += 1
            
            # Categories
            for category in paper.subject_categories:
                stats['papers_by_category'][category] += 1
                stats['categories_distribution'][category] += 1
            
            # Keywords
            for keyword in paper.keywords:
                stats['keyword_frequency'][keyword] += 1
            
            # Coherence score
            if paper.coherence_score is not None:
                coherence_scores.append(paper.coherence_score)
            
            # Date range
            if paper.publication_date:
                date = paper.publication_date
                if not stats['date_range']['earliest'] or date < stats['date_range']['earliest']:
                    stats['date_range']['earliest'] = date
                if not stats['date_range']['latest'] or date > stats['date_range']['latest']:
                    stats['date_range']['latest'] = date
            
            # Feature statistics
            if paper.energy_metrics and paper.energy_metrics.get('has_energy_discussion'):
                stats['energy_metrics_stats']['papers_with_energy_discussion'] += 1
            if paper.geometric_params and paper.geometric_params.get('has_geometry_discussion'):
                stats['geometric_params_stats']['papers_with_geometry_discussion'] += 1
        
        # Calculate average coherence score
        if coherence_scores:
            stats['avg_coherence_score'] = sum(coherence_scores) / len(coherence_scores)
        
        # Convert defaultdict to regular dict for JSON serialization
        stats['papers_by_source'] = dict(stats['papers_by_source'])
        stats['papers_by_category'] = dict(stats['papers_by_category'])
        stats['keyword_frequency'] = dict(stats['keyword_frequency'])
        stats['categories_distribution'] = dict(stats['categories_distribution'])
        
        return stats
