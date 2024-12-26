"""
System Health Monitoring Dashboard

Provides real-time monitoring and visualization of system health metrics including:
- Component performance metrics
- Resource utilization
- System bottlenecks
- Error rates and patterns
- Quantum system metrics
- Cross-component correlations
- Advanced predictive analytics
- Trend analysis
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any, Optional

from src.integration.performance_monitoring import PerformanceMonitor
from src.monitoring.predictive_analytics import SystemPredictor

class SystemHealthDashboard:
    """Real-time system health monitoring dashboard."""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        """Initialize dashboard with monitors."""
        self.performance_monitor = performance_monitor
        self.predictor = SystemPredictor()
        self.app = dash.Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
        # Initialize metrics history
        self.metrics_history = {
            'cpu_usage': [],
            'memory_usage': [],
            'latency': [],
            'error_rate': []
        }
    
    def setup_layout(self):
        """Configure dashboard layout."""
        self.app.layout = html.Div([
            html.H1('System Health Dashboard'),
            
            # Advanced Predictions
            html.Div([
                html.H2('Advanced Predictions'),
                dcc.Tabs([
                    dcc.Tab(label='Resource Predictions', children=[
                        dcc.Graph(id='resource-predictions'),
                        dcc.Interval(
                            id='resource-predictions-update',
                            interval=15000
                        )
                    ]),
                    dcc.Tab(label='Component Health', children=[
                        dcc.Graph(id='health-predictions'),
                        dcc.Interval(
                            id='health-predictions-update',
                            interval=15000
                        )
                    ]),
                    dcc.Tab(label='Anomaly Detection', children=[
                        dcc.Graph(id='anomaly-detection'),
                        dcc.Interval(
                            id='anomaly-detection-update',
                            interval=15000
                        )
                    ]),
                    dcc.Tab(label='Performance Trends', children=[
                        dcc.Graph(id='performance-trends'),
                        dcc.Interval(
                            id='performance-trends-update',
                            interval=15000
                        )
                    ])
                ])
            ]),
            
            # System Overview
            html.Div([
                html.H2('System Overview'),
                dcc.Graph(id='system-metrics'),
                dcc.Interval(
                    id='system-metrics-update',
                    interval=5000
                )
            ]),
            
            # Component Performance
            html.Div([
                html.H2('Component Performance'),
                dcc.Dropdown(
                    id='component-selector',
                    options=[
                        {'label': 'Quantum Graph', 'value': 'quantum_graph'},
                        {'label': 'Semantic Engine', 'value': 'semantic_engine'},
                        {'label': 'Profile Engine', 'value': 'profile_engine'},
                        {'label': 'Reasoning Engine', 'value': 'reasoning_engine'}
                    ],
                    value='quantum_graph'
                ),
                dcc.Graph(id='component-metrics'),
                dcc.Interval(
                    id='component-metrics-update',
                    interval=5000
                )
            ]),
            
            # Quantum System Metrics
            html.Div([
                html.H2('Quantum System Metrics'),
                dcc.Graph(id='quantum-metrics'),
                dcc.Interval(
                    id='quantum-metrics-update',
                    interval=5000
                )
            ]),
            
            # Cross-Component Correlations
            html.Div([
                html.H2('Cross-Component Correlations'),
                dcc.Graph(id='correlation-matrix'),
                dcc.Interval(
                    id='correlation-update',
                    interval=10000
                )
            ]),
            
            # Predictive Analytics
            html.Div([
                html.H2('Performance Predictions'),
                dcc.Graph(id='performance-predictions'),
                dcc.Interval(
                    id='prediction-update',
                    interval=15000
                )
            ]),
            
            # Resource Utilization
            html.Div([
                html.H2('Resource Utilization'),
                dcc.Graph(id='resource-metrics'),
                dcc.Interval(
                    id='resource-metrics-update',
                    interval=5000
                )
            ]),
            
            # Bottleneck Analysis
            html.Div([
                html.H2('Bottleneck Analysis'),
                dcc.Graph(id='bottleneck-metrics'),
                dcc.Interval(
                    id='bottleneck-metrics-update',
                    interval=10000
                )
            ]),
            
            # Error Tracking
            html.Div([
                html.H2('Error Tracking'),
                dcc.Graph(id='error-metrics'),
                dcc.Interval(
                    id='error-metrics-update',
                    interval=10000
                )
            ]),
            
            # System Health Score
            html.Div([
                html.H2('System Health Score'),
                dcc.Graph(id='health-score'),
                dcc.Interval(
                    id='health-score-update',
                    interval=5000
                )
            ])
        ])
    
    def setup_callbacks(self):
        """Configure dashboard callbacks."""
        @self.app.callback(
            Output('resource-predictions', 'figure'),
            Input('resource-predictions-update', 'n_intervals')
        )
        def update_resource_predictions(_):
            """Update resource prediction visualizations."""
            # Get historical data
            metrics = self.performance_monitor.get_system_metrics()
            
            # Update metrics history
            for metric, values in metrics.items():
                if metric in self.metrics_history:
                    self.metrics_history[metric].extend(values)
                    # Keep last 1000 points
                    self.metrics_history[metric] = self.metrics_history[metric][-1000:]
            
            # Get predictions from multiple models
            predictions = self.predictor.predict(self.metrics_history)
            
            fig = go.Figure()
            
            # Plot historical data
            fig.add_trace(go.Scatter(
                y=self.metrics_history['cpu_usage'],
                name='Historical CPU',
                mode='lines',
                line=dict(color='blue')
            ))
            
            # Plot predictions from different models
            colors = ['red', 'green', 'purple', 'orange', 'brown']
            for (model_name, result), color in zip(predictions.items(), colors):
                fig.add_trace(go.Scatter(
                    y=result.predictions,
                    name=f'{model_name} Prediction',
                    mode='lines',
                    line=dict(
                        color=color,
                        dash='dash'
                    )
                ))
                
                # Add confidence intervals if available
                if result.confidence_intervals is not None:
                    fig.add_trace(go.Scatter(
                        y=result.confidence_intervals[:, 0],
                        name=f'{model_name} Lower CI',
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        y=result.confidence_intervals[:, 1],
                        name=f'{model_name} Upper CI',
                        mode='lines',
                        line=dict(width=0),
                        fillcolor=color,
                        fill='tonexty',
                        showlegend=False
                    ))
            
            fig.update_layout(
                title='Resource Usage Predictions',
                xaxis_title='Time Point',
                yaxis_title='Usage',
                hovermode='x unified'
            )
            
            return fig
        
        @self.app.callback(
            Output('health-predictions', 'figure'),
            Input('health-predictions-update', 'n_intervals')
        )
        def update_health_predictions(_):
            """Update component health predictions."""
            # Get component metrics
            components = ['quantum_graph', 'semantic_engine',
                        'profile_engine', 'reasoning_engine']
            
            component_metrics = {}
            for component in components:
                metrics = self.performance_monitor.get_component_metrics(component)
                component_metrics[component] = np.array([
                    metrics['latency']['avg'],
                    metrics['memory_mb'],
                    metrics['cpu_percent'],
                    metrics.get('error_count', 0)
                ])
            
            # Predict health scores
            health_scores = self.predictor.predict_component_health(component_metrics)
            
            fig = go.Figure()
            
            # Create gauge charts for each component
            for component, score in health_scores.items():
                fig.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    title={'text': component},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "red"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    },
                    domain={'row': 0, 'column': components.index(component)}
                ))
            
            fig.update_layout(
                grid={'rows': 1, 'columns': len(components)},
                title='Component Health Predictions'
            )
            
            return fig
        
        @self.app.callback(
            Output('anomaly-detection', 'figure'),
            Input('anomaly-detection-update', 'n_intervals')
        )
        def update_anomaly_detection(_):
            """Update anomaly detection visualization."""
            # Get predictions with anomaly scores
            predictions = self.predictor.predict(self.metrics_history)
            
            fig = go.Figure()
            
            # Plot metrics with anomaly scores
            for metric, values in self.metrics_history.items():
                anomaly_scores = predictions['random_forest'].anomaly_scores
                
                # Normalize anomaly scores to 0-1
                normalized_scores = (anomaly_scores - np.min(anomaly_scores)) / (
                    np.max(anomaly_scores) - np.min(anomaly_scores)
                )
                
                # Color points based on anomaly score
                colors = [
                    f'rgb({int(255*s)}, {int(255*(1-s))}, 0)'
                    for s in normalized_scores
                ]
                
                fig.add_trace(go.Scatter(
                    y=values,
                    mode='markers',
                    name=metric,
                    marker=dict(
                        color=colors,
                        size=10,
                        showscale=True,
                        colorbar=dict(
                            title='Anomaly Score'
                        )
                    )
                ))
            
            fig.update_layout(
                title='Anomaly Detection',
                xaxis_title='Time Point',
                yaxis_title='Value',
                hovermode='x unified'
            )
            
            return fig
        
        @self.app.callback(
            Output('performance-trends', 'figure'),
            Input('performance-trends-update', 'n_intervals')
        )
        def update_performance_trends(_):
            """Update performance trend visualization."""
            # Analyze performance trends
            trend_analysis = self.predictor.analyze_performance_trends(
                self.metrics_history
            )
            
            fig = go.Figure()
            
            # Plot trends for each metric
            for metric, analysis in trend_analysis.items():
                if not metric.endswith('_anomalies'):
                    # Plot mean and standard deviation
                    fig.add_trace(go.Scatter(
                        y=[analysis['mean']] * len(self.metrics_history[metric]),
                        name=f'{metric} Mean',
                        mode='lines',
                        line=dict(dash='dash')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        y=np.array([analysis['mean'] + analysis['std']] *
                                 len(self.metrics_history[metric])),
                        name=f'{metric} Upper Bound',
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        y=np.array([analysis['mean'] - analysis['std']] *
                                 len(self.metrics_history[metric])),
                        name=f'{metric} Lower Bound',
                        mode='lines',
                        line=dict(width=0),
                        fillcolor='rgba(0,100,80,0.2)',
                        fill='tonexty',
                        showlegend=False
                    ))
                    
                    # Plot trend line
                    x = np.arange(len(self.metrics_history[metric]))
                    trend = analysis['trend'] * x + analysis['mean']
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=trend,
                        name=f'{metric} Trend',
                        mode='lines',
                        line=dict(
                            color='red',
                            width=2
                        )
                    ))
            
            fig.update_layout(
                title='Performance Trends Analysis',
                xaxis_title='Time Point',
                yaxis_title='Value',
                hovermode='x unified'
            )
            
            return fig
        
        @self.app.callback(
            Output('system-metrics', 'figure'),
            Input('system-metrics-update', 'n_intervals')
        )
        def update_system_metrics(_):
            """Update system metrics visualization."""
            metrics = self.performance_monitor.get_system_metrics()
            
            # Create time series for memory and CPU
            time_points = pd.date_range(
                end=datetime.now(),
                periods=len(metrics['total_memory']),
                freq='5S'
            )
            
            fig = go.Figure()
            
            # Memory usage
            fig.add_trace(go.Scatter(
                x=time_points,
                y=metrics['total_memory'],
                name='Memory Usage (MB)',
                mode='lines'
            ))
            
            # CPU usage
            fig.add_trace(go.Scatter(
                x=time_points,
                y=metrics['cpu_usage'],
                name='CPU Usage (%)',
                mode='lines'
            ))
            
            fig.update_layout(
                title='System Resource Usage',
                xaxis_title='Time',
                yaxis_title='Usage',
                hovermode='x unified'
            )
            
            return fig
        
        @self.app.callback(
            Output('component-metrics', 'figure'),
            [Input('component-metrics-update', 'n_intervals'),
             Input('component-selector', 'value')]
        )
        def update_component_metrics(_, component_id):
            """Update component-specific metrics."""
            metrics = self.performance_monitor.get_component_metrics(component_id)
            
            fig = go.Figure()
            
            # Latency metrics
            fig.add_trace(go.Scatter(
                x=['avg', 'p95', 'p99'],
                y=[
                    metrics['latency']['avg'],
                    metrics['latency']['p95'],
                    metrics['latency']['p99']
                ],
                name='Latency (ms)'
            ))
            
            fig.update_layout(
                title=f'{component_id} Performance Metrics',
                xaxis_title='Metric',
                yaxis_title='Value',
                hovermode='x'
            )
            
            return fig
        
        @self.app.callback(
            Output('quantum-metrics', 'figure'),
            Input('quantum-metrics-update', 'n_intervals')
        )
        def update_quantum_metrics(_):
            """Update quantum system metrics visualization."""
            metrics = self.performance_monitor.get_component_metrics('quantum_graph')
            
            fig = go.Figure()
            
            # Quantum state fidelity
            if 'quantum_metrics' in metrics.get('custom_metrics', {}):
                quantum_metrics = metrics['custom_metrics']['quantum_metrics']
                fig.add_trace(go.Scatter(
                    y=quantum_metrics.get('state_fidelity', []),
                    name='State Fidelity',
                    mode='lines'
                ))
                
                # Entanglement strength
                fig.add_trace(go.Scatter(
                    y=quantum_metrics.get('entanglement_strength', []),
                    name='Entanglement Strength',
                    mode='lines'
                ))
            
            fig.update_layout(
                title='Quantum System Performance',
                yaxis_title='Metric Value',
                hovermode='x unified'
            )
            
            return fig
        
        @self.app.callback(
            Output('correlation-matrix', 'figure'),
            Input('correlation-update', 'n_intervals')
        )
        def update_correlation_matrix(_):
            """Update cross-component correlation visualization."""
            components = ['quantum_graph', 'semantic_engine',
                        'profile_engine', 'reasoning_engine']
            
            # Collect metrics
            metrics_data = {}
            for component in components:
                metrics = self.performance_monitor.get_component_metrics(component)
                metrics_data[component] = {
                    'latency': metrics['latency']['avg'],
                    'memory': metrics['memory_mb'],
                    'cpu': metrics['cpu_percent']
                }
            
            # Create correlation matrix
            df = pd.DataFrame(metrics_data).T
            corr_matrix = df.corr()
            
            fig = px.imshow(
                corr_matrix,
                labels=dict(color="Correlation"),
                title="Component Metric Correlations"
            )
            
            return fig
        
        @self.app.callback(
            Output('performance-predictions', 'figure'),
            Input('prediction-update', 'n_intervals')
        )
        def update_predictions(_):
            """Update performance predictions visualization."""
            # Get historical data
            metrics = self.performance_monitor.get_system_metrics()
            
            # Prepare data for prediction
            X = np.array(range(len(metrics['cpu_usage']))).reshape(-1, 1)
            y = np.array(metrics['cpu_usage'])
            
            if len(X) > 1:  # Need at least 2 points for prediction
                # Scale data
                X_scaled = self.scaler.fit_transform(X)
                
                # Train model
                self.predictor.fit(X_scaled, y)
                
                # Predict next 10 points
                future_X = np.array(range(len(X), len(X) + 10)).reshape(-1, 1)
                future_X_scaled = self.scaler.transform(future_X)
                predictions = self.predictor.predict(future_X_scaled)
                
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=X.flatten(),
                    y=y,
                    name='Historical',
                    mode='lines'
                ))
                
                # Predictions
                fig.add_trace(go.Scatter(
                    x=future_X.flatten(),
                    y=predictions,
                    name='Predicted',
                    mode='lines',
                    line=dict(dash='dash')
                ))
                
                fig.update_layout(
                    title='CPU Usage Predictions',
                    xaxis_title='Time Point',
                    yaxis_title='CPU Usage (%)'
                )
                
                return fig
            
            return go.Figure()
        
        @self.app.callback(
            Output('resource-metrics', 'figure'),
            Input('resource-metrics-update', 'n_intervals')
        )
        def update_resource_metrics(_):
            """Update resource utilization metrics."""
            metrics = self.performance_monitor.get_system_metrics()
            
            fig = go.Figure()
            
            # Memory distribution
            fig.add_trace(go.Box(
                y=metrics['total_memory'],
                name='Memory Usage (MB)'
            ))
            
            # CPU distribution
            fig.add_trace(go.Box(
                y=metrics['cpu_usage'],
                name='CPU Usage (%)'
            ))
            
            fig.update_layout(
                title='Resource Utilization Distribution',
                yaxis_title='Usage',
                hovermode='x'
            )
            
            return fig
        
        @self.app.callback(
            Output('bottleneck-metrics', 'figure'),
            Input('bottleneck-metrics-update', 'n_intervals')
        )
        def update_bottleneck_metrics(_):
            """Update bottleneck analysis visualization."""
            report = self.performance_monitor.generate_performance_report()
            bottlenecks = report['bottlenecks']
            
            if not bottlenecks:
                return go.Figure()
            
            df = pd.DataFrame(bottlenecks)
            
            fig = px.bar(
                df,
                x='component',
                y='avg_latency',
                title='Component Bottlenecks',
                labels={
                    'component': 'Component',
                    'avg_latency': 'Average Latency (ms)'
                }
            )
            
            return fig
        
        @self.app.callback(
            Output('error-metrics', 'figure'),
            Input('error-metrics-update', 'n_intervals')
        )
        def update_error_metrics(_):
            """Update error tracking visualization."""
            components = ['quantum_graph', 'semantic_engine',
                        'profile_engine', 'reasoning_engine']
            error_counts = []
            
            for component in components:
                metrics = self.performance_monitor.get_component_metrics(component)
                error_counts.append(metrics.get('error_count', 0))
            
            fig = go.Figure(data=[
                go.Bar(
                    x=components,
                    y=error_counts,
                    name='Error Count'
                )
            ])
            
            fig.update_layout(
                title='Component Error Counts',
                xaxis_title='Component',
                yaxis_title='Number of Errors'
            )
            
            return fig
        
        @self.app.callback(
            Output('health-score', 'figure'),
            Input('health-score-update', 'n_intervals')
        )
        def update_health_score(_):
            """Update system health score visualization."""
            # Collect metrics
            system_metrics = self.performance_monitor.get_system_metrics()
            
            # Calculate health scores
            cpu_health = max(0, 100 - np.mean(system_metrics['cpu_usage']))
            memory_health = max(0, 100 - (
                np.mean(system_metrics['total_memory']) /
                self.performance_monitor.config['memory_warning_threshold_mb'] * 100
            ))
            
            # Component health
            component_health = []
            for component in ['quantum_graph', 'semantic_engine',
                            'profile_engine', 'reasoning_engine']:
                metrics = self.performance_monitor.get_component_metrics(component)
                if metrics:
                    # Score based on latency and error rate
                    latency_score = max(0, 100 - (
                        metrics['latency']['avg'] /
                        self.performance_monitor.config['bottleneck_threshold_ms'] * 100
                    ))
                    error_score = max(0, 100 - (
                        metrics['error_count'] / max(metrics['call_count'], 1) * 100
                    ))
                    component_health.append((latency_score + error_score) / 2)
            
            # Overall health score
            overall_health = np.mean([
                cpu_health,
                memory_health,
                *component_health
            ])
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=overall_health,
                title={'text': "System Health Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            return fig
    
    def run_server(self, debug: bool = False, port: int = 8050):
        """Start the dashboard server."""
        self.app.run_server(debug=debug, port=port)
