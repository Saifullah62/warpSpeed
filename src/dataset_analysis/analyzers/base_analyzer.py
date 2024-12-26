from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from datasets import Dataset
import pandas as pd
import numpy as np
from datetime import datetime
import logging

class BaseAnalyzer(ABC):
    """Base class for dataset analyzers."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics_history: List[Dict[str, Any]] = []
        
    @abstractmethod
    def analyze(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Analyze the dataset and return metrics.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        pass
        
    @abstractmethod
    def detect_anomalies(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Detect anomalies in the dataset.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary containing detected anomalies
        """
        pass
        
    def track_metrics(self, metrics: Dict[str, Any]):
        """
        Track metrics over time.
        
        Args:
            metrics: Dictionary of metrics to track
        """
        timestamp = datetime.now()
        self.metrics_history.append({
            'timestamp': timestamp,
            'metrics': metrics
        })
        
    def get_metrics_history(self) -> pd.DataFrame:
        """
        Get historical metrics as a DataFrame.
        
        Returns:
            DataFrame containing metrics history
        """
        if not self.metrics_history:
            return pd.DataFrame()
            
        records = []
        for entry in self.metrics_history:
            record = {'timestamp': entry['timestamp']}
            record.update(entry['metrics'])
            records.append(record)
            
        return pd.DataFrame(records)
        
    def detect_drift(self, 
                    current_metrics: Dict[str, Any],
                    threshold: float = 0.1) -> Dict[str, Any]:
        """
        Detect significant changes in metrics.
        
        Args:
            current_metrics: Current metrics
            threshold: Threshold for significant change
            
        Returns:
            Dictionary containing drift detection results
        """
        if not self.metrics_history:
            return {}
            
        previous_metrics = self.metrics_history[-1]['metrics']
        drift_detected = {}
        
        for metric, current_value in current_metrics.items():
            if metric in previous_metrics:
                previous_value = previous_metrics[metric]
                if isinstance(current_value, (int, float)) and \
                   isinstance(previous_value, (int, float)):
                    change = abs(current_value - previous_value) / max(abs(previous_value), 1)
                    if change > threshold:
                        drift_detected[metric] = {
                            'previous': previous_value,
                            'current': current_value,
                            'change': change
                        }
                        
        return drift_detected
        
    def summarize_analysis(self, 
                          analysis_results: Dict[str, Any],
                          anomalies: Dict[str, Any],
                          drift: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a comprehensive analysis summary.
        
        Args:
            analysis_results: Results from analyze()
            anomalies: Results from detect_anomalies()
            drift: Results from detect_drift()
            
        Returns:
            Dictionary containing analysis summary
        """
        return {
            'timestamp': datetime.now(),
            'analysis_results': analysis_results,
            'anomalies_detected': anomalies,
            'drift_detected': drift,
            'metrics_tracked': len(self.metrics_history)
        }
