import os
from src.analysis.advanced_concepts_analyzer import AdvancedConceptsAnalyzer

def main():
    # Get the absolute path to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Set up paths
    data_dir = os.path.join(project_root, 'data')
    output_dir = os.path.join(project_root, 'reports')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    print("Initializing Advanced Concepts Analysis...")
    analyzer = AdvancedConceptsAnalyzer(data_dir, output_dir)
    
    # Generate report
    print("Analyzing advanced warp drive concepts...")
    report_path = analyzer.generate_advanced_concepts_report()
    print(f"\nAdvanced concepts report has been generated: {report_path}")
    print("\nKey sections in the report:")
    print("1. Research Progress Analysis")
    print("2. Technical Breakthroughs Analysis")
    print("3. Integration Opportunities Analysis")
    print("4. Development Challenges Analysis")
    print("5. Research Recommendations")
    print("6. Next Steps")

if __name__ == "__main__":
    main()
