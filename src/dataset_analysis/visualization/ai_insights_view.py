import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, List
import json

class AIInsightsView:
    """Visualization component for AI-powered analysis insights."""
    
    def show_ai_insights(self, results_df: pd.DataFrame):
        """Display AI analysis insights."""
        if results_df.empty:
            st.warning("No AI analysis data available.")
            return
            
        st.header("AI-Powered Analysis Insights")
        
        # Show latest analysis
        latest = results_df.iloc[-1]
        
        # Create tabs for different aspects
        tabs = st.tabs([
            "Consensus Metrics",
            "Claude Analysis",
            "GPT-4 Analysis",
            "Enhanced Metrics"
        ])
        
        with tabs[0]:
            self._show_consensus_metrics(latest)
            
        with tabs[1]:
            self._show_claude_analysis(latest)
            
        with tabs[2]:
            self._show_gpt4_analysis(latest)
            
        with tabs[3]:
            self._show_enhanced_metrics(latest)
            
    def _show_consensus_metrics(self, latest: pd.Series):
        """Show consensus metrics from both AI models."""
        st.subheader("Consensus Metrics")
        
        metrics = latest.get('consensus_metrics', {})
        if isinstance(metrics, str):
            metrics = json.loads(metrics)
            
        cols = st.columns(3)
        
        # Create gauge charts for each metric
        with cols[0]:
            self._create_gauge_chart(
                "Innovation Potential",
                metrics.get('innovation_potential', 0)
            )
            
        with cols[1]:
            self._create_gauge_chart(
                "Research Maturity",
                metrics.get('research_maturity', 0)
            )
            
        with cols[2]:
            self._create_gauge_chart(
                "Implementation Feasibility",
                metrics.get('implementation_feasibility', 0)
            )
            
    def _show_claude_analysis(self, latest: pd.Series):
        """Show Claude's analysis results."""
        st.subheader("Claude Analysis")
        
        claude_analysis = latest.get('claude_analysis', {})
        if isinstance(claude_analysis, str):
            claude_analysis = json.loads(claude_analysis)
            
        # Show themes and insights
        cols = st.columns(2)
        
        with cols[0]:
            st.write("**Key Themes**")
            themes = claude_analysis.get('themes', [])
            for theme in themes:
                st.write(f"- {theme}")
                
        with cols[1]:
            st.write("**Key Insights**")
            insights = claude_analysis.get('insights', [])
            for insight in insights:
                st.write(f"- {insight}")
                
        # Show complexity analysis
        st.write("**Complexity Analysis**")
        complexity = claude_analysis.get('complexity_levels', 0)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=complexity,
            title={'text': "Technical Complexity"},
            gauge={'axis': {'range': [0, 10]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 3], 'color': "lightgray"},
                       {'range': [3, 7], 'color': "gray"},
                       {'range': [7, 10], 'color': "darkgray"}
                   ]}
        ))
        st.plotly_chart(fig)
        
    def _show_gpt4_analysis(self, latest: pd.Series):
        """Show GPT-4's analysis results."""
        st.subheader("GPT-4 Analysis")
        
        gpt4_analysis = latest.get('gpt4_analysis', {})
        if isinstance(gpt4_analysis, str):
            gpt4_analysis = json.loads(gpt4_analysis)
            
        # Show innovation score
        innovation_score = gpt4_analysis.get('innovation_scores', 0)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=innovation_score,
            title={'text': "Innovation Score"},
            gauge={'axis': {'range': [0, 10]},
                   'bar': {'color': "darkred"},
                   'steps': [
                       {'range': [0, 3], 'color': "lightgray"},
                       {'range': [3, 7], 'color': "gray"},
                       {'range': [7, 10], 'color': "darkgray"}
                   ]}
        ))
        st.plotly_chart(fig)
        
        # Show research gaps and future directions
        cols = st.columns(2)
        
        with cols[0]:
            st.write("**Research Gaps**")
            gaps = gpt4_analysis.get('research_gaps', [])
            for gap in gaps:
                st.write(f"- {gap}")
                
        with cols[1]:
            st.write("**Future Directions**")
            directions = gpt4_analysis.get('future_directions', [])
            for direction in directions:
                st.write(f"- {direction}")
                
    def _show_enhanced_metrics(self, latest: pd.Series):
        """Show enhanced metrics combining both analyses."""
        st.subheader("Enhanced Metrics")
        
        enhanced_metrics = latest.get('enhanced_metrics', {})
        if isinstance(enhanced_metrics, str):
            enhanced_metrics = json.loads(enhanced_metrics)
            
        # Show confidence scores
        confidence = enhanced_metrics.get('analysis_confidence', {})
        
        # Create confidence comparison chart
        fig = go.Figure(data=[
            go.Bar(
                x=['AI Confidence', 'Traditional Confidence', 'Combined Confidence'],
                y=[
                    confidence.get('ai_confidence', 0),
                    confidence.get('traditional_confidence', 0),
                    confidence.get('combined_confidence', 0)
                ],
                marker_color=['rgb(55, 83, 109)', 'rgb(26, 118, 255)', 'rgb(15, 133, 84)']
            )
        ])
        
        fig.update_layout(
            title="Analysis Confidence Comparison",
            yaxis_title="Confidence Score",
            showlegend=False
        )
        
        st.plotly_chart(fig)
        
        # Show topic coverage
        topic_coverage = enhanced_metrics.get('topic_coverage', {})
        if topic_coverage:
            st.write("**Topic Coverage Analysis**")
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=['Overlapping Topics', 'Unique Traditional', 'Unique AI'],
                    values=[
                        topic_coverage.get('overlap', 0),
                        topic_coverage.get('unique_traditional', 0),
                        topic_coverage.get('unique_ai', 0)
                    ],
                    hole=.3
                )
            ])
            
            fig.update_layout(title="Topic Distribution")
            st.plotly_chart(fig)
            
    def _create_gauge_chart(self, title: str, value: float):
        """Create a gauge chart for metrics."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title},
            gauge={'axis': {'range': [0, 10]},
                   'bar': {'color': "darkblue"},
                   'steps': [
                       {'range': [0, 3], 'color': "lightgray"},
                       {'range': [3, 7], 'color': "gray"},
                       {'range': [7, 10], 'color': "darkgray"}
                   ]}
        ))
        st.plotly_chart(fig)
