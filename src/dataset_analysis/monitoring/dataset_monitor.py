from typing import Dict, Any, Optional, List, Type
from datasets import Dataset
import schedule
import time
import logging
from datetime import datetime
import json
from pathlib import Path
import threading
from queue import Queue
import pandas as pd
import asyncio

from ..analyzers.base_analyzer import BaseAnalyzer
from ..analyzers.ai_powered_analyzer import AIPoweredAnalyzer
from ..core.dataset_loader import DatasetLoader

class DatasetMonitor:
    """Monitors datasets for changes and runs continuous analysis."""
    
    def __init__(self, 
                 dataset_loader: DatasetLoader,
                 analyzers: Dict[str, BaseAnalyzer],
                 storage_dir: str,
                 alert_threshold: float = 0.1):
        self.dataset_loader = dataset_loader
        self.analyzers = analyzers
        self.storage_dir = Path(storage_dir)
        self.alert_threshold = alert_threshold
        self.logger = logging.getLogger(__name__)
        self.analysis_queue = Queue()
        self.should_stop = False
        
        # Create storage directory if it doesn't exist
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    def start_monitoring(self, 
                        dataset_name: str,
                        interval_minutes: int = 60,
                        **dataset_kwargs):
        """
        Start monitoring a dataset.
        
        Args:
            dataset_name: Name of the dataset to monitor
            interval_minutes: Monitoring interval in minutes
            **dataset_kwargs: Additional arguments for dataset loading
        """
        def monitoring_job():
            try:
                # Load dataset
                dataset = self.dataset_loader.load_dataset(
                    dataset_name, **dataset_kwargs
                )
                
                # Queue analysis task
                self.analysis_queue.put({
                    'dataset': dataset,
                    'timestamp': datetime.now(),
                    'dataset_name': dataset_name
                })
                
            except Exception as e:
                self.logger.error(f"Error in monitoring job: {str(e)}")
        
        # Schedule regular monitoring
        schedule.every(interval_minutes).minutes.do(monitoring_job)
        
        # Start worker thread for analysis
        worker_thread = threading.Thread(target=self._analysis_worker)
        worker_thread.start()
        
        # Run scheduling loop
        while not self.should_stop:
            schedule.run_pending()
            time.sleep(1)
            
        worker_thread.join()
        
    def stop_monitoring(self):
        """Stop all monitoring activities."""
        self.should_stop = True
        
    def _analysis_worker(self):
        """Worker thread for processing analysis queue."""
        while not self.should_stop:
            try:
                # Get analysis task from queue
                if self.analysis_queue.empty():
                    time.sleep(1)
                    continue
                    
                task = self.analysis_queue.get()
                dataset = task['dataset']
                timestamp = task['timestamp']
                dataset_name = task['dataset_name']
                
                # Run analysis with each analyzer
                for analyzer_name, analyzer in self.analyzers.items():
                    try:
                        # Run analysis
                        analysis_results = analyzer.analyze(dataset)
                        anomalies = analyzer.detect_anomalies(dataset)
                        drift = analyzer.detect_drift(analysis_results)
                        
                        # Create summary
                        summary = analyzer.summarize_analysis(
                            analysis_results, anomalies, drift
                        )
                        
                        # Save results
                        self._save_results(
                            dataset_name, analyzer_name, timestamp, summary
                        )
                        
                        # Check for alerts
                        self._check_alerts(
                            dataset_name, analyzer_name, drift
                        )
                        
                    except Exception as e:
                        self.logger.error(
                            f"Error in analyzer {analyzer_name}: {str(e)}"
                        )
                        
            except Exception as e:
                self.logger.error(f"Error in analysis worker: {str(e)}")
                
    def _save_results(self,
                     dataset_name: str,
                     analyzer_name: str,
                     timestamp: datetime,
                     results: Dict[str, Any]):
        """Save analysis results to storage."""
        try:
            # Create result directory
            result_dir = self.storage_dir / dataset_name / analyzer_name
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            result_file = result_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, default=str)
                
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            
    def _check_alerts(self,
                     dataset_name: str,
                     analyzer_name: str,
                     drift: Dict[str, Any]):
        """Check for alert conditions."""
        if drift:
            alert_message = (
                f"Dataset: {dataset_name}\n"
                f"Analyzer: {analyzer_name}\n"
                "Significant changes detected:\n"
            )
            
            for metric, details in drift.items():
                alert_message += (
                    f"- {metric}: changed by "
                    f"{details['change']*100:.2f}% "
                    f"({details['previous']} → {details['current']})\n"
                )
                
            self.logger.warning(alert_message)
            
    def get_historical_metrics(self,
                             dataset_name: str,
                             analyzer_name: str,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Retrieve historical metrics for a dataset and analyzer.
        
        Args:
            dataset_name: Name of the dataset
            analyzer_name: Name of the analyzer
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            DataFrame containing historical metrics
        """
        result_dir = self.storage_dir / dataset_name / analyzer_name
        if not result_dir.exists():
            return pd.DataFrame()
            
        # Load all result files
        results = []
        for result_file in result_dir.glob('*.json'):
            try:
                # Parse timestamp from filename
                timestamp = datetime.strptime(
                    result_file.stem, '%Y%m%d_%H%M%S'
                )
                
                # Apply date filtering
                if start_date and timestamp < start_date:
                    continue
                if end_date and timestamp > end_date:
                    continue
                    
                # Load results
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    data['timestamp'] = timestamp
                    results.append(data)
                    
            except Exception as e:
                self.logger.error(
                    f"Error loading result file {result_file}: {str(e)}"
                )
                
        return pd.DataFrame(results)
