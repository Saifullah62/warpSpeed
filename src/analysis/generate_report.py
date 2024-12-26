import os
from src.analysis.report_generator import ReportGenerator

def main():
    # Get the absolute path to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Set up paths
    data_dir = os.path.join(project_root, 'data')
    output_dir = os.path.join(project_root, 'reports')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize report generator
    report_generator = ReportGenerator(data_dir, output_dir)
    
    # Generate report
    print("Generating research analysis report...")
    report_path = report_generator.generate_report()
    print(f"\nReport has been generated: {report_path}")

if __name__ == "__main__":
    main()
