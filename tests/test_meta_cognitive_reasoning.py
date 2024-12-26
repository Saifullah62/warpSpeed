import pytest
import numpy as np

from src.reasoning.meta_cognitive_reasoning_engine import (
    MetaCognitiveReasoningEngine,
    ReasoningState
)
from src.reasoning.causal_reasoning_engine import CausalReasoningEngine
from src.reasoning.abductive_reasoning_engine import AbductiveReasoningEngine

class TestMetaCognitiveReasoningEngine:
    @pytest.fixture
    def meta_cognitive_reasoning_engine(self):
        """Create a meta-cognitive reasoning engine"""
        causal_engine = CausalReasoningEngine()
        abductive_engine = AbductiveReasoningEngine()
        return MetaCognitiveReasoningEngine(causal_engine, abductive_engine)
    
    def test_reasoning_process_initialization(self, meta_cognitive_reasoning_engine):
        """
        Test reasoning process initialization
        
        Validates:
        - Reasoning state creation
        - Initial state properties
        """
        initial_context = {
            'domain': 'technology',
            'research_area': 'quantum computing'
        }
        
        # Initialize reasoning process
        reasoning_state = meta_cognitive_reasoning_engine.initiate_reasoning_process(
            initial_context
        )
        
        # Validate reasoning state
        assert isinstance(reasoning_state, ReasoningState), "Invalid reasoning state type"
        
        # Check initial state properties
        assert reasoning_state.reasoning_depth == 0, "Incorrect initial reasoning depth"
        assert reasoning_state.uncertainty_level == 1.0, "Incorrect initial uncertainty level"
        assert reasoning_state.cognitive_complexity == 0.0, "Incorrect initial cognitive complexity"
        assert reasoning_state.reasoning_strategy == 'default', "Incorrect initial reasoning strategy"
        assert len(reasoning_state.explored_hypotheses) == 0, "Explored hypotheses should be empty"
        assert len(reasoning_state.reasoning_history) > 0, "Reasoning history should have initialization entry"
    
    def test_adaptive_reasoning_strategy(self, meta_cognitive_reasoning_engine):
        """
        Test adaptive reasoning strategy selection
        
        Validates:
        - Strategy selection mechanism
        - Strategy selection factors
        """
        # Create initial reasoning state
        reasoning_state = ReasoningState(
            reasoning_depth=2,
            uncertainty_level=0.7,
            cognitive_complexity=0.5,
            reasoning_strategy='default'
        )
        
        # Test contexts
        test_contexts = [
            {'domain': 'technology', 'complexity': 'high'},
            {'domain': 'science', 'complexity': 'low'}
        ]
        
        for context in test_contexts:
            # Select reasoning strategy
            strategy = meta_cognitive_reasoning_engine.adaptive_reasoning_strategy(
                reasoning_state, 
                context
            )
            
            # Validate strategy
            assert strategy in ['causal', 'abductive', 'exploratory'], f"Invalid strategy for context {context}"
    
    def test_iterative_reasoning(self, meta_cognitive_reasoning_engine):
        """
        Test iterative reasoning process
        
        Validates:
        - Reasoning iteration mechanism
        - State updates during reasoning
        """
        # Initialize reasoning process
        initial_context = {
            'domain': 'technology',
            'research_area': 'quantum computing'
        }
        
        reasoning_state = meta_cognitive_reasoning_engine.initiate_reasoning_process(
            initial_context
        )
        reasoning_id = list(meta_cognitive_reasoning_engine.reasoning_states.keys())[0]
        
        # Perform iterative reasoning
        iteration_results = meta_cognitive_reasoning_engine.reason_iteratively(
            reasoning_id, 
            max_iterations=3
        )
        
        # Validate iteration results
        assert isinstance(iteration_results, list), "Invalid iteration results type"
        assert 1 <= len(iteration_results) <= 3, "Incorrect number of iterations"
        
        # Check individual iteration results
        for result in iteration_results:
            assert 'strategy' in result, "Missing strategy in iteration result"
            assert 'results' in result, "Missing results in iteration result"
            assert 'complexity' in result, "Missing complexity in iteration result"
            
            # Validate strategy
            assert result['strategy'] in ['causal', 'abductive', 'exploratory'], "Invalid reasoning strategy"
    
    def test_reasoning_trajectory_analysis(self, meta_cognitive_reasoning_engine):
        """
        Test reasoning trajectory analysis
        
        Validates:
        - Trajectory analysis computation
        - Trajectory metrics
        """
        # Initialize reasoning process
        initial_context = {
            'domain': 'science',
            'research_area': 'complex systems'
        }
        
        reasoning_state = meta_cognitive_reasoning_engine.initiate_reasoning_process(
            initial_context
        )
        reasoning_id = list(meta_cognitive_reasoning_engine.reasoning_states.keys())[0]
        
        # Perform iterative reasoning
        meta_cognitive_reasoning_engine.reason_iteratively(
            reasoning_id, 
            max_iterations=3
        )
        
        # Analyze reasoning trajectory
        trajectory_analysis = meta_cognitive_reasoning_engine.analyze_reasoning_trajectory(
            reasoning_id
        )
        
        # Validate trajectory analysis
        assert isinstance(trajectory_analysis, dict), "Invalid trajectory analysis type"
        
        # Check trajectory metrics
        assert 'total_depth' in trajectory_analysis, "Missing total depth in trajectory analysis"
        assert 'final_uncertainty' in trajectory_analysis, "Missing final uncertainty in trajectory analysis"
        assert 'final_complexity' in trajectory_analysis, "Missing final complexity in trajectory analysis"
        assert 'explored_hypotheses_count' in trajectory_analysis, "Missing explored hypotheses count in trajectory analysis"
        assert 'strategies_used' in trajectory_analysis, "Missing strategies used in trajectory analysis"
        
        # Validate metric ranges
        assert 0 <= trajectory_analysis['total_depth'] <= 5, "Invalid total reasoning depth"
        assert 0 <= trajectory_analysis['final_uncertainty'] <= 1, "Invalid final uncertainty"
        assert 0 <= trajectory_analysis['final_complexity'] <= 1, "Invalid final complexity"
        
        # Check strategies
        assert all(
            strategy in ['causal', 'abductive', 'exploratory'] 
            for strategy in trajectory_analysis['strategies_used']
        ), "Invalid reasoning strategies"
    
    def test_reasoning_state_update(self, meta_cognitive_reasoning_engine):
        """
        Test reasoning state update mechanism
        
        Validates:
        - State update logic
        - Uncertainty and complexity changes
        """
        # Create initial reasoning state
        reasoning_state = ReasoningState(
            reasoning_depth=0,
            uncertainty_level=1.0,
            cognitive_complexity=0.0,
            reasoning_strategy='default'
        )
        
        # Simulate iteration results
        test_iterations = [
            {
                'strategy': 'causal',
                'results': [{'path': ['A', 'B']}],
                'complexity': 0.3
            },
            {
                'strategy': 'abductive',
                'results': [{'id': 'hyp1'}, {'id': 'hyp2'}],
                'complexity': 0.5
            }
        ]
        
        for iteration_result in test_iterations:
            # Update reasoning state
            meta_cognitive_reasoning_engine._update_reasoning_state(
                reasoning_state, 
                iteration_result
            )
        
        # Validate state updates
        assert reasoning_state.reasoning_depth == 0, "Reasoning depth should not change"
        assert reasoning_state.uncertainty_level < 1.0, "Uncertainty should decrease"
        assert reasoning_state.cognitive_complexity > 0.0, "Cognitive complexity should increase"
        
        if test_iterations[-1]['strategy'] == 'abductive':
            assert len(reasoning_state.explored_hypotheses) > 0, "Explored hypotheses should be tracked"
        
        assert len(reasoning_state.reasoning_history) > 1, "Reasoning history should be updated"
