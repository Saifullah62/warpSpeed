import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AlertConfig:
    threshold: float
    window_size: int = 10
    alert_cooldown: timedelta = timedelta(minutes=5)

class MetricsMonitor:
    def __init__(self):
        self.metrics_history: Dict[str, List[float]] = {}
        self.alert_configs: Dict[str, AlertConfig] = {
            'quantum_stability': AlertConfig(threshold=0.9),
            'energy_density': AlertConfig(threshold=1e15),
            'field_uniformity': AlertConfig(threshold=0.85),
            'coherence': AlertConfig(threshold=0.8)
        }
        self.last_alert_time: Dict[str, datetime] = {}
        
    async def monitor_metrics(self, metrics: Dict[str, float]):
        """Monitor metrics in real-time and generate alerts if needed."""
        try:
            for metric_name, value in metrics.items():
                if metric_name not in self.metrics_history:
                    self.metrics_history[metric_name] = []
                    
                self.metrics_history[metric_name].append(value)
                
                # Keep only recent history
                if len(self.metrics_history[metric_name]) > 1000:
                    self.metrics_history[metric_name] = self.metrics_history[metric_name][-1000:]
                    
                # Check for anomalies
                if metric_name in self.alert_configs:
                    await self._check_anomalies(metric_name)
                    
        except Exception as e:
            logger.error(f"Error monitoring metrics: {str(e)}")
            
    async def _check_anomalies(self, metric_name: str):
        """Check for anomalies in metric values."""
        try:
            config = self.alert_configs[metric_name]
            history = self.metrics_history[metric_name]
            
            if len(history) < config.window_size:
                return
                
            recent_values = history[-config.window_size:]
            
            # Calculate statistics
            mean_value = np.mean(recent_values)
            std_value = np.std(recent_values)
            
            # Check for threshold violations
            if mean_value > config.threshold:
                await self._generate_alert(
                    metric_name,
                    f"{metric_name} exceeded threshold: {mean_value:.2f} > {config.threshold:.2f}"
                )
                
            # Check for sudden changes
            if std_value > 0.1 * mean_value:
                await self._generate_alert(
                    metric_name,
                    f"High variability detected in {metric_name}: std={std_value:.2f}, mean={mean_value:.2f}"
                )
                
        except Exception as e:
            logger.error(f"Error checking anomalies: {str(e)}")
            
    async def _generate_alert(self, metric_name: str, message: str):
        """Generate an alert if cooldown period has passed."""
        now = datetime.now()
        
        if (metric_name not in self.last_alert_time or
            now - self.last_alert_time[metric_name] > self.alert_configs[metric_name].alert_cooldown):
            
            logger.warning(f"ALERT: {message}")
            # Here you could add additional alert channels (email, Slack, etc.)
            
            self.last_alert_time[metric_name] = now
            
    async def get_metric_summary(self, metric_name: str) -> Dict[str, float]:
        """Get statistical summary of a metric's recent history."""
        try:
            if metric_name not in self.metrics_history:
                return {}
                
            history = self.metrics_history[metric_name]
            if not history:
                return {}
                
            return {
                'current_value': history[-1],
                'mean': np.mean(history),
                'std': np.std(history),
                'min': np.min(history),
                'max': np.max(history),
                'samples': len(history)
            }
            
        except Exception as e:
            logger.error(f"Error generating metric summary: {str(e)}")
            return {}
