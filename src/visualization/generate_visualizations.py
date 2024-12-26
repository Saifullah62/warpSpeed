import os
from src.visualization.data_visualizer import DataVisualizer

def main():
    # Get the absolute path to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Set up paths
    data_dir = os.path.join(project_root, 'data')
    output_dir = os.path.join(project_root, 'visualizations')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = DataVisualizer(data_dir, output_dir)
    
    # Generate all visualizations
    print("Generating visualizations...")
    visualizer.generate_all_visualizations()
    print("Visualizations have been generated in:", output_dir)
    print("\nVisualization files:")
    print("1. concept_network.png - Network of related research concepts")
    print("2. technology_heatmap.png - Heatmap showing technology relationships")
    print("3. research_timeline.html - Interactive timeline of research developments")
    print("4. topic_clusters.png - Clustering of research topics")
    print("5. wordcloud.png - Word cloud of key concepts")

if __name__ == "__main__":
    main()
