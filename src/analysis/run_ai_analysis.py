import os
from src.dataset_analysis.core.dataset_loader import DatasetLoader
from src.dataset_analysis.analyzers.ai_powered_analyzer import AIPoweredAnalyzer
from src.dataset_analysis.monitoring.dataset_monitor import DatasetMonitor

def main():
    # Get the absolute path to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Set up paths
    data_dir = os.path.join(project_root, 'data')
    output_dir = os.path.join(project_root, 'reports')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    print("Initializing AI-Powered Analysis System...")
    dataset_loader = DatasetLoader()
    ai_analyzer = AIPoweredAnalyzer(text_column='summary')
    
    # Set up monitoring with AI analyzer
    monitor = DatasetMonitor(
        dataset_loader=dataset_loader,
        analyzers={'ai_analyzer': ai_analyzer},
        storage_dir=output_dir
    )
    
    # Load sample dataset
    print("\nLoading dataset...")
    dataset_path = os.path.join(data_dir, 'sample_research_data.csv')
    dataset = dataset_loader.load_dataset('csv', data_files=dataset_path)
    
    # Run analysis
    print("\nRunning AI-powered analysis...")
    analysis_results = ai_analyzer.analyze(dataset)
    
    print("\nAnalysis Results:")
    print("\nClaude Analysis:")
    print("- Themes:", len(analysis_results['claude_analysis']['themes']))
    print("- Insights:", len(analysis_results['claude_analysis']['insights']))
    print("- Complexity Level:", analysis_results['claude_analysis']['complexity_levels'])
    
    print("\nGPT-4 Analysis:")
    print("- Innovation Score:", analysis_results['gpt4_analysis']['innovation_scores'])
    print("- Research Gaps:", len(analysis_results['gpt4_analysis']['research_gaps']))
    print("- Future Directions:", len(analysis_results['gpt4_analysis']['future_directions']))
    
    print("\nConsensus Metrics:")
    print("- Innovation Potential:", analysis_results['consensus_metrics']['innovation_potential'])
    print("- Research Maturity:", analysis_results['consensus_metrics']['research_maturity'])
    print("- Implementation Feasibility:", analysis_results['consensus_metrics']['implementation_feasibility'])
    
    # Check for anomalies
    print("\nChecking for anomalies...")
    anomalies = ai_analyzer.detect_anomalies(dataset)
    
    if anomalies:
        print("\nAnomalies Detected:")
        for anomaly_type, issues in anomalies.items():
            print(f"\n{anomaly_type}:")
            for issue in issues:
                print(f"- {issue}")
    else:
        print("No significant anomalies detected.")
    
    print("\nAnalysis complete! Results have been saved to:", output_dir)

if __name__ == "__main__":
    main()
