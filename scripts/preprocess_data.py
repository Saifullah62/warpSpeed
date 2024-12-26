"""
Script to run the data preprocessing pipeline.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_processing.data_preprocessor import DataPreprocessor
import json

def main():
    # Initialize paths
    base_dir = project_root
    data_dir = base_dir / "data"
    output_dir = base_dir / "processed_data"
    output_dir.mkdir(exist_ok=True)

    # Initialize preprocessor
    preprocessor = DataPreprocessor(data_dir)

    print("Loading papers...")
    # Load papers from different sources
    preprocessor.load_arxiv_papers()
    preprocessor.load_nasa_papers()
    
    print(f"Loaded {len(preprocessor.papers)} papers")

    print("Computing features...")
    # Compute additional features
    preprocessor.compute_features()

    print("Removing duplicates...")
    # Remove duplicate papers
    preprocessor.remove_duplicates()

    print("Filtering irrelevant papers...")
    # Filter out irrelevant papers
    preprocessor.filter_irrelevant()

    print("Generating statistics...")
    # Generate and save statistics
    stats = preprocessor.generate_statistics()
    with open(output_dir / "dataset_statistics.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    print("Saving processed data...")
    # Save processed data
    preprocessor.save_processed_data(output_dir / "processed_papers.json")

    print("\nDataset Statistics:")
    print(f"Total papers: {stats['total_papers']}")
    print("\nPapers by source:")
    for source, count in stats['papers_by_source'].items():
        print(f"  {source}: {count}")
    print(f"\nDate range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
    print(f"Average coherence score: {stats['avg_coherence_score']:.2f}")

    print("\nDone! Processed data saved to:", output_dir)

if __name__ == "__main__":
    main()
