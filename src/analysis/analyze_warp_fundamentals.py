import os
from src.analysis.warp_fundamentals_analyzer import WarpFundamentalsAnalyzer

def main():
    # Get the absolute path to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Set up paths
    data_dir = os.path.join(project_root, 'data')
    output_dir = os.path.join(project_root, 'reports')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    print("Initializing Warp Drive Fundamentals Analysis...")
    analyzer = WarpFundamentalsAnalyzer(data_dir, output_dir)
    
    # Generate report
    print("Analyzing warp drive fundamentals...")
    report_path = analyzer.generate_warp_fundamentals_report()
    print(f"\nWarp drive fundamentals report has been generated: {report_path}")
    print("\nKey sections in the report:")
    print("1. Core Components Analysis")
    print("2. Theoretical Foundations Analysis")
    print("3. Technical Challenges Analysis")
    print("4. Development Roadmap Analysis")
    print("5. Key Findings and Implications")
    print("6. Next Steps")

if __name__ == "__main__":
    main()
