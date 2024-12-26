import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import numpy as np
from datetime import datetime

def load_dataset_stats():
    """Load the latest dataset statistics."""
    metadata_dir = Path("metadata")
    stats_files = list(metadata_dir.glob("dataset_stats_*.json"))
    latest_stats_file = max(stats_files, key=lambda x: x.stat().st_mtime)
    
    with open(latest_stats_file, 'r') as f:
        return json.load(f)

def create_category_distribution(stats, output_dir):
    """Create a horizontal bar chart of paper categories."""
    categories = pd.DataFrame.from_dict(
        stats["papers_by_category"], 
        orient='index',
        columns=['count']
    ).sort_values('count', ascending=True)
    
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    sns.barplot(x=categories['count'], y=categories.index, palette='viridis')
    
    plt.title('Distribution of Papers by Category', pad=20, size=14)
    plt.xlabel('Number of Papers')
    plt.ylabel('Category')
    
    # Add value labels on the bars
    for i, v in enumerate(categories['count']):
        plt.text(v + 1, i, str(v), va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'category_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_version_distribution(stats, output_dir):
    """Create a pie chart of paper versions."""
    versions = pd.DataFrame.from_dict(
        stats["versions_distribution"], 
        orient='index',
        columns=['count']
    )
    
    plt.figure(figsize=(10, 10))
    colors = sns.color_palette('viridis', n_colors=len(versions))
    
    plt.pie(versions['count'], labels=[f'v{i} ({v})' for i, v in versions['count'].items()],
            autopct='%1.1f%%', colors=colors, startangle=90)
    
    plt.title('Distribution of Paper Versions', pad=20, size=14)
    plt.axis('equal')
    
    plt.savefig(output_dir / 'version_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_category_version_heatmap(stats, output_dir):
    """Create a heatmap showing the relationship between categories and versions."""
    # Load the full papers metadata to create the heatmap
    with open(Path("metadata") / "papers_metadata.json", 'r') as f:
        papers = json.load(f)
    
    # Create a DataFrame with category and version information
    df = pd.DataFrame([{
        'category': paper['category'],
        'version': paper['version']
    } for paper in papers])
    
    # Create a pivot table for the heatmap
    pivot = pd.crosstab(df['category'], df['version'])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='d', cmap='viridis')
    
    plt.title('Category vs Version Distribution', pad=20, size=14)
    plt.xlabel('Version')
    plt.ylabel('Category')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'category_version_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_stats(stats, output_dir):
    """Create a summary statistics visualization."""
    plt.figure(figsize=(10, 6))
    
    summary_data = {
        'Total Papers': stats['total_papers'],
        'New Papers': stats['processing_summary']['new_papers'],
        'Updated Papers': stats['processing_summary']['updated_papers']
    }
    
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    plt.bar(summary_data.keys(), summary_data.values(), color=colors)
    
    plt.title('Dataset Summary Statistics', pad=20, size=14)
    plt.ylabel('Number of Papers')
    
    # Add value labels on top of bars
    for i, v in enumerate(summary_data.values()):
        plt.text(i, v + 1, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_stats.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create visualizations directory if it doesn't exist
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Load statistics
    stats = load_dataset_stats()
    
    # Create visualizations
    print("Creating category distribution plot...")
    create_category_distribution(stats, output_dir)
    
    print("Creating version distribution plot...")
    create_version_distribution(stats, output_dir)
    
    print("Creating category-version heatmap...")
    create_category_version_heatmap(stats, output_dir)
    
    print("Creating summary statistics plot...")
    create_summary_stats(stats, output_dir)
    
    print(f"Visualizations have been saved to {output_dir}")

if __name__ == "__main__":
    main()
