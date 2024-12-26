import os
from src.visualization.research_trends_visualizer import ResearchTrendsVisualizer

def main():
    # Get the absolute path to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Set up paths
    data_dir = os.path.join(project_root, 'data')
    output_dir = os.path.join(project_root, 'reports')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    print("Initializing Research Trends Visualizer...")
    visualizer = ResearchTrendsVisualizer(data_dir, output_dir)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    viz_paths = visualizer.generate_all_visualizations()
    
    print("\nVisualizations have been generated:")
    print("1. Research Timeline (Interactive)")
    print("2. Concept Network")
    print("3. Breakthrough Heatmap")
    print("4. Technology Readiness Assessment (Interactive)")
    print("5. Research Focus Treemap (Interactive)")
    
    print("\nVisualization files:")
    for path in viz_paths:
        print(f"- {path}")

if __name__ == "__main__":
    main()
