import os
from src.analysis.development_roadmap_analyzer import DevelopmentRoadmapAnalyzer

def main():
    # Get the absolute path to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Set up paths
    data_dir = os.path.join(project_root, 'data')
    output_dir = os.path.join(project_root, 'reports')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    print("Initializing Development Roadmap Analysis...")
    analyzer = DevelopmentRoadmapAnalyzer(data_dir, output_dir)
    
    # Generate roadmap
    print("Generating development roadmap...")
    report_path = analyzer.generate_development_roadmap()
    print(f"\nDevelopment roadmap has been generated: {report_path}")
    print("\nKey deliverables generated:")
    print("1. Detailed development roadmap (development_roadmap.md)")
    print("2. Task dependency network visualization (task_network.png)")
    print("3. Interactive development timeline (development_timeline.html)")
    
if __name__ == "__main__":
    main()
