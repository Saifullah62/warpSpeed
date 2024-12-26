import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
from pathlib import Path
from .ai_insights_view import AIInsightsView

class AnalysisDashboard:
    """Interactive dashboard for visualizing dataset analysis results."""
    
    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.ai_insights = AIInsightsView()
        
    def run(self):
        """Run the dashboard application."""
        st.title("Dataset Analysis Dashboard")
        
        # Sidebar for controls
        self._create_sidebar()
        
        # Main content tabs
        tabs = st.tabs([
            "Overview",
            "AI Insights",
            "Detailed Analysis",
            "Anomalies",
            "Drift Analysis"
        ])
        
        results_df = self._load_results()
        
        with tabs[0]:
            self._show_overview(results_df)
            
        with tabs[1]:
            self.ai_insights.show_ai_insights(results_df)
            
        with tabs[2]:
            self._show_detailed_analysis(results_df)
            
        with tabs[3]:
            self._show_anomalies(results_df)
            
        with tabs[4]:
            self._show_drift_analysis(results_df)
        
    def _create_sidebar(self):
        """Create sidebar with control options."""
        st.sidebar.header("Controls")
        
        # Dataset selection
        available_datasets = [d.name for d in self.storage_dir.iterdir() 
                            if d.is_dir()]
        selected_dataset = st.sidebar.selectbox(
            "Select Dataset",
            available_datasets
        )
        
        # Analyzer selection
        if selected_dataset:
            dataset_dir = self.storage_dir / selected_dataset
            available_analyzers = [d.name for d in dataset_dir.iterdir() 
                                 if d.is_dir()]
            selected_analyzer = st.sidebar.selectbox(
                "Select Analyzer",
                available_analyzers
            )
        
        # Time range selection
        st.sidebar.subheader("Time Range")
        time_range = st.sidebar.radio(
            "Select Time Range",
            ["Last 24 Hours", "Last Week", "Last Month", "Custom"]
        )
        
        if time_range == "Custom":
            end_date = st.sidebar.date_input(
                "End Date",
                datetime.now()
            )
            start_date = st.sidebar.date_input(
                "Start Date",
                end_date - timedelta(days=7)
            )
        else:
            end_date = datetime.now()
            if time_range == "Last 24 Hours":
                start_date = end_date - timedelta(days=1)
            elif time_range == "Last Week":
                start_date = end_date - timedelta(days=7)
            else:  # Last Month
                start_date = end_date - timedelta(days=30)
        
        # Store selections in session state
        st.session_state.selected_dataset = selected_dataset
        st.session_state.selected_analyzer = selected_analyzer
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        
    def _load_results(self) -> pd.DataFrame:
        """Load analysis results based on current selections."""
        if not hasattr(st.session_state, 'selected_dataset') or \
           not hasattr(st.session_state, 'selected_analyzer'):
            return pd.DataFrame()
            
        result_dir = (self.storage_dir / 
                     st.session_state.selected_dataset / 
                     st.session_state.selected_analyzer)
        
        results = []
        for result_file in result_dir.glob('*.json'):
            try:
                timestamp = datetime.strptime(
                    result_file.stem, '%Y%m%d_%H%M%S'
                )
                
                if (timestamp.date() >= st.session_state.start_date and
                    timestamp.date() <= st.session_state.end_date):
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        data['timestamp'] = timestamp
                        results.append(data)
                        
            except Exception as e:
                st.error(f"Error loading result file {result_file}: {str(e)}")
                
        return pd.DataFrame(results)
        
    def _show_overview(self, results_df: pd.DataFrame):
        """Show overview metrics and trends."""
        st.header("Overview")
        
        if results_df.empty:
            st.warning("No data available for the selected criteria.")
            return
            
        # Create metrics columns
        cols = st.columns(3)
        
        # Show latest metrics
        latest_results = results_df.iloc[-1]
        
        if 'analysis_results' in latest_results:
            metrics = latest_results['analysis_results']
            if isinstance(metrics, str):
                metrics = json.loads(metrics)
                
            # Display key metrics
            for i, (metric, value) in enumerate(metrics.items()):
                if isinstance(value, (int, float)):
                    cols[i % 3].metric(
                        metric,
                        f"{value:.2f}",
                        self._calculate_change(results_df, metric)
                    )
                    
        # Show trends
        st.subheader("Trends")
        self._plot_trends(results_df)
        
    def _show_detailed_analysis(self, results_df: pd.DataFrame):
        """Show detailed analysis results."""
        st.header("Detailed Analysis")
        
        if results_df.empty:
            return
            
        # Show latest analysis results
        st.subheader("Latest Analysis")
        latest = results_df.iloc[-1]
        
        if 'analysis_results' in latest:
            results = latest['analysis_results']
            if isinstance(results, str):
                results = json.loads(results)
                
            for metric, value in results.items():
                if isinstance(value, dict):
                    st.write(f"**{metric}**")
                    st.json(value)
                elif isinstance(value, list):
                    st.write(f"**{metric}**")
                    for item in value:
                        st.write(f"- {item}")
                else:
                    st.write(f"**{metric}**: {value}")
                    
    def _show_anomalies(self, results_df: pd.DataFrame):
        """Show detected anomalies."""
        st.header("Anomalies")
        
        if results_df.empty:
            return
            
        # Show latest anomalies
        latest = results_df.iloc[-1]
        
        if 'anomalies_detected' in latest:
            anomalies = latest['anomalies_detected']
            if isinstance(anomalies, str):
                anomalies = json.loads(anomalies)
                
            if anomalies:
                for anomaly_type, details in anomalies.items():
                    st.subheader(f"{anomaly_type}")
                    st.json(details)
            else:
                st.success("No anomalies detected in the latest analysis.")
                
    def _show_drift_analysis(self, results_df: pd.DataFrame):
        """Show drift analysis results."""
        st.header("Drift Analysis")
        
        if results_df.empty:
            return
            
        # Show latest drift
        latest = results_df.iloc[-1]
        
        if 'drift_detected' in latest:
            drift = latest['drift_detected']
            if isinstance(drift, str):
                drift = json.loads(drift)
                
            if drift:
                for metric, details in drift.items():
                    st.subheader(f"{metric} Drift")
                    
                    # Create drift visualization
                    fig = go.Figure()
                    fig.add_trace(go.Indicator(
                        mode="number+delta",
                        value=details['current'],
                        delta={'reference': details['previous'],
                               'relative': True},
                        title={'text': metric}
                    ))
                    
                    st.plotly_chart(fig)
            else:
                st.success("No significant drift detected in the latest analysis.")
                
    def _plot_trends(self, results_df: pd.DataFrame):
        """Plot trend charts for metrics."""
        if 'analysis_results' not in results_df.columns:
            return
            
        # Extract metrics from analysis results
        metrics_data = []
        for _, row in results_df.iterrows():
            results = row['analysis_results']
            if isinstance(results, str):
                results = json.loads(results)
                
            metrics = {
                'timestamp': row['timestamp']
            }
            
            for metric, value in results.items():
                if isinstance(value, (int, float)):
                    metrics[metric] = value
                    
            metrics_data.append(metrics)
            
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create trend charts
        for column in metrics_df.columns:
            if column != 'timestamp' and \
               metrics_df[column].dtype in ['int64', 'float64']:
                fig = px.line(
                    metrics_df,
                    x='timestamp',
                    y=column,
                    title=f"{column} Trend"
                )
                st.plotly_chart(fig)
                
    def _calculate_change(self,
                         results_df: pd.DataFrame,
                         metric: str) -> float:
        """Calculate change in metric from previous measurement."""
        if len(results_df) < 2:
            return None
            
        latest = results_df.iloc[-1]
        previous = results_df.iloc[-2]
        
        if 'analysis_results' in latest and 'analysis_results' in previous:
            latest_results = latest['analysis_results']
            previous_results = previous['analysis_results']
            
            if isinstance(latest_results, str):
                latest_results = json.loads(latest_results)
            if isinstance(previous_results, str):
                previous_results = json.loads(previous_results)
                
            if metric in latest_results and metric in previous_results:
                latest_value = latest_results[metric]
                previous_value = previous_results[metric]
                
                if isinstance(latest_value, (int, float)) and \
                   isinstance(previous_value, (int, float)):
                    return (latest_value - previous_value) / previous_value
                    
        return None
