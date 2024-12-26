"""
Performance Monitoring System for Star Trek Technology Development

This module provides comprehensive performance monitoring capabilities including:
- Latency tracking across all system components
- Memory usage monitoring and optimization
- Computational overhead analysis
- System bottleneck detection
- Resource utilization metrics
- Component-specific performance profiling

The monitoring system uses async collectors to minimize overhead and provides
both real-time metrics and historical analysis capabilities.
"""

import time
import asyncio
import psutil
import tracemalloc
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging
import numpy as np
from contextlib import contextmanager

@dataclass
class ComponentMetrics:
    """
    Stores performance metrics for a specific component
    
    Attributes:
        component_id: Unique identifier for the component
        latency_ms: List of latency measurements in milliseconds
        memory_mb: Memory usage in megabytes
        cpu_percent: CPU utilization percentage
        gpu_utilization: GPU utilization if available
        call_count: Number of times the component was called
        error_count: Number of errors encountered
        last_bottleneck: Timestamp of last detected bottleneck
        custom_metrics: Component-specific metrics
    """
    component_id: str
    latency_ms: List[float]
    memory_mb: float
    cpu_percent: float
    gpu_utilization: Optional[float]
    call_count: int
    error_count: int
    last_bottleneck: Optional[datetime]
    custom_metrics: Dict[str, Any]

class PerformanceMonitor:
    """
    Main performance monitoring system that tracks system-wide metrics
    and provides analysis capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the performance monitor
        
        Args:
            config: Configuration dictionary with monitoring parameters
        """
        self.config = config or {
            'sampling_rate_ms': 100,
            'history_size': 1000,
            'bottleneck_threshold_ms': 100,
            'memory_warning_threshold_mb': 1000,
            'enable_gpu_monitoring': True
        }
        
        # Initialize trackers
        self.metrics: Dict[str, ComponentMetrics] = {}
        self.system_metrics = {
            'total_memory': [],
            'cpu_usage': [],
            'disk_io': [],
            'network_io': []
        }
        
        # Start memory tracking
        tracemalloc.start()
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    async def start_monitoring(self):
        """Start the async monitoring loops"""
        await asyncio.gather(
            self._monitor_system_metrics(),
            self._monitor_component_metrics(),
            self._analyze_bottlenecks()
        )
    
    @contextmanager
    def component_timer(self, component_id: str):
        """
        Context manager for timing component execution
        
        Args:
            component_id: Unique identifier for the component
        
        Example:
            with monitor.component_timer('semantic_engine'):
                result = semantic_engine.process(input_data)
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            self._record_latency(component_id, (end_time - start_time) * 1000)
    
    def _record_latency(self, component_id: str, latency_ms: float):
        """
        Record component latency
        
        Args:
            component_id: Component identifier
            latency_ms: Latency in milliseconds
        """
        if component_id not in self.metrics:
            self.metrics[component_id] = ComponentMetrics(
                component_id=component_id,
                latency_ms=[],
                memory_mb=0.0,
                cpu_percent=0.0,
                gpu_utilization=None,
                call_count=0,
                error_count=0,
                last_bottleneck=None,
                custom_metrics={}
            )
        
        metrics = self.metrics[component_id]
        metrics.latency_ms.append(latency_ms)
        metrics.call_count += 1
        
        # Check for bottleneck
        if latency_ms > self.config['bottleneck_threshold_ms']:
            metrics.last_bottleneck = datetime.now()
            self.logger.warning(
                f"Performance bottleneck detected in {component_id}: "
                f"{latency_ms:.2f}ms"
            )
    
    async def _monitor_system_metrics(self):
        """Monitor system-wide metrics"""
        while True:
            # Memory usage
            memory = psutil.Process().memory_info()
            self.system_metrics['total_memory'].append(memory.rss / 1024 / 1024)
            
            # CPU usage
            self.system_metrics['cpu_usage'].append(psutil.cpu_percent())
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            self.system_metrics['disk_io'].append({
                'read_bytes': disk_io.read_bytes,
                'write_bytes': disk_io.write_bytes
            })
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.system_metrics['network_io'].append({
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            })
            
            # Check memory threshold
            if (self.system_metrics['total_memory'][-1] >
                self.config['memory_warning_threshold_mb']):
                self.logger.warning(
                    f"Memory usage exceeds threshold: "
                    f"{self.system_metrics['total_memory'][-1]:.2f}MB"
                )
            
            await asyncio.sleep(self.config['sampling_rate_ms'] / 1000)
    
    async def _monitor_component_metrics(self):
        """Monitor component-specific metrics"""
        while True:
            snapshot = tracemalloc.take_snapshot()
            for component_id, metrics in self.metrics.items():
                # Update memory usage
                component_memory = sum(
                    stat.size for stat in snapshot.statistics('filename')
                    if component_id in stat.traceback[0].filename
                )
                metrics.memory_mb = component_memory / 1024 / 1024
                
                # Update CPU usage
                metrics.cpu_percent = psutil.Process().cpu_percent()
                
                # Update GPU utilization if available
                try:
                    if self.config['enable_gpu_monitoring']:
                        # Add GPU monitoring implementation here
                        pass
                except Exception as e:
                    self.logger.warning(f"GPU monitoring failed: {str(e)}")
            
            await asyncio.sleep(self.config['sampling_rate_ms'] / 1000)
    
    async def _analyze_bottlenecks(self):
        """Analyze system bottlenecks"""
        while True:
            for component_id, metrics in self.metrics.items():
                if len(metrics.latency_ms) > 0:
                    # Calculate performance metrics
                    avg_latency = np.mean(metrics.latency_ms)
                    p95_latency = np.percentile(metrics.latency_ms, 95)
                    p99_latency = np.percentile(metrics.latency_ms, 99)
                    
                    # Log performance issues
                    if p99_latency > self.config['bottleneck_threshold_ms']:
                        self.logger.warning(
                            f"Performance degradation in {component_id}:\n"
                            f"Avg: {avg_latency:.2f}ms, "
                            f"P95: {p95_latency:.2f}ms, "
                            f"P99: {p99_latency:.2f}ms"
                        )
            
            await asyncio.sleep(1.0)  # Analyze every second
    
    def get_component_metrics(
        self,
        component_id: str,
        metric_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get metrics for a specific component
        
        Args:
            component_id: Component identifier
            metric_type: Optional specific metric type to retrieve
        
        Returns:
            Dictionary of component metrics
        """
        if component_id not in self.metrics:
            return {}
        
        metrics = self.metrics[component_id]
        if metric_type:
            if metric_type == 'latency':
                return {
                    'avg_latency': np.mean(metrics.latency_ms),
                    'p95_latency': np.percentile(metrics.latency_ms, 95),
                    'p99_latency': np.percentile(metrics.latency_ms, 99)
                }
            elif metric_type == 'memory':
                return {'memory_mb': metrics.memory_mb}
            elif metric_type == 'cpu':
                return {'cpu_percent': metrics.cpu_percent}
            elif metric_type == 'calls':
                return {
                    'call_count': metrics.call_count,
                    'error_count': metrics.error_count
                }
        
        return {
            'latency': {
                'avg': np.mean(metrics.latency_ms),
                'p95': np.percentile(metrics.latency_ms, 95),
                'p99': np.percentile(metrics.latency_ms, 99)
            },
            'memory_mb': metrics.memory_mb,
            'cpu_percent': metrics.cpu_percent,
            'gpu_utilization': metrics.gpu_utilization,
            'call_count': metrics.call_count,
            'error_count': metrics.error_count,
            'last_bottleneck': metrics.last_bottleneck,
            'custom_metrics': metrics.custom_metrics
        }
    
    def get_system_metrics(
        self,
        metric_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get system-wide metrics
        
        Args:
            metric_type: Optional specific metric type to retrieve
        
        Returns:
            Dictionary of system metrics
        """
        if metric_type:
            return self.system_metrics.get(metric_type, {})
        return self.system_metrics
    
    def add_custom_metric(
        self,
        component_id: str,
        metric_name: str,
        value: Any
    ):
        """
        Add a custom metric for a component
        
        Args:
            component_id: Component identifier
            metric_name: Name of the custom metric
            value: Metric value
        """
        if component_id in self.metrics:
            self.metrics[component_id].custom_metrics[metric_name] = value
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report
        
        Returns:
            Dictionary containing the performance report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': {
                'memory_usage': {
                    'current': self.system_metrics['total_memory'][-1],
                    'avg': np.mean(self.system_metrics['total_memory'])
                },
                'cpu_usage': {
                    'current': self.system_metrics['cpu_usage'][-1],
                    'avg': np.mean(self.system_metrics['cpu_usage'])
                }
            },
            'component_metrics': {},
            'bottlenecks': [],
            'recommendations': []
        }
        
        # Add component metrics
        for component_id, metrics in self.metrics.items():
            report['component_metrics'][component_id] = self.get_component_metrics(
                component_id
            )
            
            # Identify bottlenecks
            if metrics.last_bottleneck:
                report['bottlenecks'].append({
                    'component': component_id,
                    'last_occurrence': metrics.last_bottleneck,
                    'avg_latency': np.mean(metrics.latency_ms)
                })
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(
        self,
        report: Dict[str, Any]
    ) -> List[str]:
        """
        Generate performance optimization recommendations
        
        Args:
            report: Performance report dictionary
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Memory recommendations
        current_memory = report['system_metrics']['memory_usage']['current']
        if current_memory > self.config['memory_warning_threshold_mb']:
            recommendations.append(
                f"High memory usage detected ({current_memory:.2f}MB). "
                "Consider implementing memory optimization strategies."
            )
        
        # CPU recommendations
        if report['system_metrics']['cpu_usage']['avg'] > 80:
            recommendations.append(
                "High average CPU usage. Consider implementing caching or "
                "optimizing computational intensive operations."
            )
        
        # Component-specific recommendations
        for component_id, metrics in report['component_metrics'].items():
            if metrics['latency']['p99'] > self.config['bottleneck_threshold_ms']:
                recommendations.append(
                    f"High latency detected in {component_id}. "
                    "Consider optimizing or parallelizing operations."
                )
        
        return recommendations
