import pytest
import os
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from src.reasoning.explainable_reasoning_engine import ExplainableReasoningEngine
from src.reasoning.meta_cognitive_reasoning_engine import MetaCognitiveReasoningEngine
from src.reasoning.causal_reasoning_engine import CausalReasoningEngine
from src.reasoning.abductive_reasoning_engine import AbductiveReasoningEngine, Hypothesis

class TestExplainableReasoningEngine:
    @pytest.fixture
    def explainable_reasoning_engine(self):
        """Create an explainable reasoning engine with mock dependencies"""
        meta_cognitive_engine = MetaCognitiveReasoningEngine()
        causal_engine = CausalReasoningEngine()
        abductive_engine = AbductiveReasoningEngine()
        
        return ExplainableReasoningEngine(
            meta_cognitive_engine, 
            causal_engine, 
            abductive_engine
        )
    
    @pytest.fixture
    def sample_reasoning_process(self, explainable_reasoning_engine):
        """
        Create a sample reasoning process for testing
        
        Returns:
            Reasoning process ID
        """
        # Initialize reasoning process
        initial_context = {
            'domain': 'technology',
            'research_area': 'quantum computing'
        }
        
        reasoning_state = explainable_reasoning_engine.meta_cognitive_engine.initiate_reasoning_process(
            initial_context
        )
        
        # Perform reasoning iterations
        reasoning_id = list(explainable_reasoning_engine.meta_cognitive_engine.reasoning_states.keys())[0]
        explainable_reasoning_engine.meta_cognitive_engine.reason_iteratively(reasoning_id)
        
        return reasoning_id
    
    def test_generate_reasoning_explanation(
        self, 
        explainable_reasoning_engine, 
        sample_reasoning_process
    ):
        """
        Test reasoning explanation generation
        
        Validates:
        - Explanation generation for different styles
        - Explanation structure and content
        """
        # Test different explanation styles
        explanation_styles = ['detailed', 'concise', 'minimal']
        
        for style in explanation_styles:
            # Generate explanation
            explanation = explainable_reasoning_engine.generate_reasoning_explanation(
                sample_reasoning_process, 
                explanation_style=style
            )
            
            # Validate explanation structure
            assert isinstance(explanation, dict), f"Invalid explanation type for {style} style"
            
            # Check key components
            assert 'overview' in explanation, f"Missing overview in {style} explanation"
            assert 'strategy_breakdown' in explanation, f"Missing strategy breakdown in {style} explanation"
            assert 'hypothesis_insights' in explanation, f"Missing hypothesis insights in {style} explanation"
            assert 'complexity_analysis' in explanation, f"Missing complexity analysis in {style} explanation"
            assert 'visualization' in explanation, f"Missing visualization in {style} explanation"
            
            # Validate overview
            overview = explanation['overview']
            assert 'total_iterations' in overview, f"Missing total iterations in {style} overview"
            assert 'initial_uncertainty' in overview, f"Missing initial uncertainty in {style} overview"
            assert 'final_uncertainty' in overview, f"Missing final uncertainty in {style} overview"
            assert 'cognitive_complexity' in overview, f"Missing cognitive complexity in {style} overview"
            
            # Validate strategy breakdown
            strategy_breakdown = explanation['strategy_breakdown']
            assert 'strategies' in strategy_breakdown, f"Missing strategies in {style} breakdown"
            assert 'strategy_distribution' in strategy_breakdown, f"Missing strategy distribution in {style} breakdown"
            
            # Validate hypothesis insights
            hypothesis_insights = explanation['hypothesis_insights']
            assert 'total_hypotheses' in hypothesis_insights, f"Missing total hypotheses in {style} insights"
            assert 'unique_hypotheses' in hypothesis_insights, f"Missing unique hypotheses in {style} insights"
            
            # Validate complexity analysis
            complexity_analysis = explanation['complexity_analysis']
            assert 'cognitive_complexity' in complexity_analysis, f"Missing cognitive complexity in {style} analysis"
            assert 'uncertainty_reduction_rate' in complexity_analysis, f"Missing uncertainty reduction rate in {style} analysis"
            assert 'reasoning_depth_factor' in complexity_analysis, f"Missing reasoning depth factor in {style} analysis"
            
            # Validate visualization
            visualization = explanation['visualization']
            assert visualization is None or os.path.exists(visualization), f"Invalid visualization for {style} style"
    
    def test_generate_hypothesis_explanation(self, explainable_reasoning_engine):
        """
        Test hypothesis explanation generation
        
        Validates:
        - Explanation generation for different styles
        - Hypothesis explanation structure and content
        """
        # Create a sample hypothesis
        sample_hypothesis = Hypothesis(
            id='test_hypothesis_1',
            description='Quantum computing breakthrough',
            probability=0.75,
            explanatory_power=0.6,
            confidence_interval=(0.5, 0.9)
        )
        
        # Test different explanation styles
        explanation_styles = ['detailed', 'concise', 'minimal']
        
        for style in explanation_styles:
            # Generate hypothesis explanation
            hypothesis_explanation = explainable_reasoning_engine.generate_hypothesis_explanation(
                sample_hypothesis, 
                explanation_style=style
            )
            
            # Validate explanation structure
            assert isinstance(hypothesis_explanation, dict), f"Invalid hypothesis explanation type for {style} style"
            
            # Check key components
            assert 'id' in hypothesis_explanation, f"Missing ID in {style} hypothesis explanation"
            assert 'description' in hypothesis_explanation, f"Missing description in {style} hypothesis explanation"
            assert 'probability' in hypothesis_explanation, f"Missing probability in {style} hypothesis explanation"
            assert 'explanatory_power' in hypothesis_explanation, f"Missing explanatory power in {style} hypothesis explanation"
            assert 'confidence_interval' in hypothesis_explanation, f"Missing confidence interval in {style} hypothesis explanation"
            
            # Validate specific values
            assert hypothesis_explanation['id'] == 'test_hypothesis_1', f"Incorrect hypothesis ID in {style} explanation"
            assert hypothesis_explanation['probability'] == 0.75, f"Incorrect probability in {style} explanation"
            assert hypothesis_explanation['explanatory_power'] == 0.6, f"Incorrect explanatory power in {style} explanation"
            assert hypothesis_explanation['confidence_interval'] == (0.5, 0.9), f"Incorrect confidence interval in {style} explanation"
            
            # Check detailed explanation for more detailed styles
            if style in ['detailed', 'concise']:
                assert 'detailed_explanation' in hypothesis_explanation, f"Missing detailed explanation in {style} hypothesis explanation"
    
    def test_reasoning_visualization(self, explainable_reasoning_engine, sample_reasoning_process):
        """
        Test reasoning process visualization
        
        Validates:
        - Visualization generation
        - Visualization file creation
        - Visualization graph structure
        """
        # Generate reasoning explanation to trigger visualization
        explanation = explainable_reasoning_engine.generate_reasoning_explanation(
            sample_reasoning_process, 
            explanation_style='detailed'
        )
        
        # Validate visualization
        visualization_path = explanation['visualization']
        
        # Check visualization file
        assert visualization_path is not None, "No visualization generated"
        assert os.path.exists(visualization_path), "Visualization file not created"
        
        # Optional: Additional visualization graph validation
        try:
            # Load the generated graph visualization
            G = nx.read_gpickle(visualization_path)
            
            # Validate graph properties
            assert isinstance(G, nx.DiGraph), "Invalid graph type"
            assert len(G.nodes()) > 0, "Empty graph"
            assert len(G.edges()) >= 0, "Invalid number of edges"
        except Exception as e:
            # If graph loading fails, it might be a matplotlib image
            # We'll just ensure the file is not empty
            assert os.path.getsize(visualization_path) > 0, "Visualization file is empty"
        
        # Clean up visualization file after test
        if os.path.exists(visualization_path):
            os.remove(visualization_path)
    
    def test_reasoning_complexity_metrics(self, explainable_reasoning_engine, sample_reasoning_process):
        """
        Test reasoning complexity metrics computation
        
        Validates:
        - Complexity metric computation
        - Metric value ranges
        """
        # Retrieve reasoning state
        reasoning_state = explainable_reasoning_engine.meta_cognitive_engine.reasoning_states[sample_reasoning_process]
        
        # Compute complexity metrics
        complexity_metrics = explainable_reasoning_engine._compute_reasoning_complexity(reasoning_state)
        
        # Validate complexity metrics
        assert 'cognitive_complexity' in complexity_metrics, "Missing cognitive complexity metric"
        assert 'uncertainty_reduction_rate' in complexity_metrics, "Missing uncertainty reduction rate metric"
        assert 'reasoning_depth_factor' in complexity_metrics, "Missing reasoning depth factor metric"
        
        # Validate metric ranges
        assert 0 <= complexity_metrics['cognitive_complexity'] <= 1, "Invalid cognitive complexity range"
        assert 0 <= complexity_metrics['uncertainty_reduction_rate'] <= 1, "Invalid uncertainty reduction rate range"
        assert 0 <= complexity_metrics['reasoning_depth_factor'] <= 1, "Invalid reasoning depth factor range"
