import pytest
import numpy as np
import networkx as nx

from src.reasoning.abductive_reasoning_engine import (
    AbductiveReasoningEngine,
    Hypothesis
)
from src.reasoning.causal_reasoning_engine import CausalReasoningEngine

@pytest.fixture
def sample_knowledge_graph():
    """Create a sample knowledge graph for abductive reasoning"""
    G = nx.DiGraph()
    
    # Add sample nodes with domain information
    nodes = [
        {'id': 'quantum_computing', 'domain': 'technology', 'description': 'Advanced computing paradigm'},
        {'id': 'machine_learning', 'domain': 'artificial_intelligence', 'description': 'Data-driven learning algorithms'},
        {'id': 'neural_networks', 'domain': 'artificial_intelligence', 'description': 'Computational models inspired by brain structure'},
        {'id': 'quantum_algorithms', 'domain': 'technology', 'description': 'Algorithms leveraging quantum mechanics'}
    ]
    
    for node in nodes:
        G.add_node(node['id'], **node)
    
    # Add sample edges
    edges = [
        ('quantum_computing', 'quantum_algorithms'),
        ('machine_learning', 'neural_networks'),
        ('quantum_computing', 'machine_learning')
    ]
    
    G.add_edges_from(edges)
    
    return G

@pytest.fixture
def causal_reasoning_engine(sample_knowledge_graph):
    """Create a causal reasoning engine with sample knowledge graph"""
    return CausalReasoningEngine(sample_knowledge_graph)

class TestAbductiveReasoningEngine:
    @pytest.fixture
    def abductive_reasoning_engine(self, causal_reasoning_engine, sample_knowledge_graph):
        """Create an abductive reasoning engine with dependencies"""
        return AbductiveReasoningEngine(
            causal_reasoning_engine, 
            sample_knowledge_graph
        )
    
    def test_hypothesis_generation(self, abductive_reasoning_engine):
        """
        Test hypothesis generation process
        
        Validates:
        - Hypothesis generation from observations
        - Hypothesis properties
        """
        # Sample observations
        observations = [
            {
                'id': 'obs_1',
                'description': 'Unexpected quantum computing breakthrough',
                'domain': 'technology',
                'timestamp': '2024-01-01'
            },
            {
                'id': 'obs_2',
                'description': 'Novel machine learning algorithm discovered',
                'domain': 'artificial_intelligence',
                'timestamp': '2024-01-02'
            }
        ]
        
        # Generate hypotheses
        hypotheses = abductive_reasoning_engine.generate_hypotheses(observations)
        
        # Validate hypothesis generation
        assert isinstance(hypotheses, list), "Invalid hypotheses type"
        assert len(hypotheses) > 0, "No hypotheses generated"
        
        # Check individual hypothesis properties
        for hypothesis in hypotheses:
            assert isinstance(hypothesis, Hypothesis), "Invalid hypothesis type"
            
            # Validate hypothesis attributes
            assert hypothesis.id is not None, "Missing hypothesis ID"
            assert hypothesis.description is not None, "Missing hypothesis description"
            
            # Check probability and explanatory power
            assert 0 <= hypothesis.probability <= 1, "Invalid hypothesis probability"
            assert 0 <= hypothesis.explanatory_power <= 1, "Invalid explanatory power"
            
            # Check confidence interval
            lower, upper = hypothesis.confidence_interval
            assert 0 <= lower <= upper <= 1, "Invalid confidence interval"
    
    def test_hypothesis_ranking(self, abductive_reasoning_engine):
        """
        Test hypothesis ranking mechanism
        
        Validates:
        - Hypothesis ranking based on probability and explanatory power
        - Filtering of low-probability hypotheses
        """
        # Sample observations
        observations = [
            {
                'id': 'obs_1',
                'description': 'Complex technological phenomenon',
                'domain': 'technology',
                'timestamp': '2024-01-01'
            }
        ]
        
        # Generate hypotheses
        hypotheses = abductive_reasoning_engine.generate_hypotheses(observations)
        
        # Validate ranking
        assert len(hypotheses) <= 10, "Too many hypotheses generated"
        
        # Check descending order of composite score
        for i in range(1, len(hypotheses)):
            prev_hypothesis = hypotheses[i-1]
            current_hypothesis = hypotheses[i]
            
            # Compute composite scores
            prev_score = (
                abductive_reasoning_engine.explanatory_power_weight * prev_hypothesis.explanatory_power +
                (1 - abductive_reasoning_engine.explanatory_power_weight) * prev_hypothesis.probability
            )
            
            current_score = (
                abductive_reasoning_engine.explanatory_power_weight * current_hypothesis.explanatory_power +
                (1 - abductive_reasoning_engine.explanatory_power_weight) * current_hypothesis.probability
            )
            
            assert prev_score >= current_score, "Hypotheses not correctly ranked"
    
    def test_hypothesis_evaluation(self, abductive_reasoning_engine):
        """
        Test hypothesis evaluation with new evidence
        
        Validates:
        - Hypothesis update mechanism
        - Evidence relevance computation
        """
        # Sample observations and initial hypotheses
        observations = [
            {
                'id': 'obs_1',
                'description': 'Quantum computing breakthrough',
                'domain': 'technology',
                'timestamp': '2024-01-01'
            }
        ]
        
        # Generate hypotheses
        hypotheses = abductive_reasoning_engine.generate_hypotheses(observations)
        
        # Select first hypothesis for evaluation
        hypothesis = hypotheses[0]
        
        # New evidence
        new_evidence = {
            'description': 'Additional quantum computing research',
            'domain': 'technology',
            'confidence': 0.8
        }
        
        # Evaluate hypothesis
        updated_hypothesis = abductive_reasoning_engine.evaluate_hypothesis(
            hypothesis.id, 
            new_evidence
        )
        
        # Validate hypothesis update
        assert updated_hypothesis.id == hypothesis.id, "Hypothesis ID changed"
        
        # Check evidence tracking
        assert len(updated_hypothesis.supporting_evidence) > 0, "No supporting evidence added"
        
        # Verify probability and confidence interval changes
        assert updated_hypothesis.probability != hypothesis.probability, "Probability not updated"
        
        lower, upper = updated_hypothesis.confidence_interval
        assert 0 <= lower <= upper <= 1, "Invalid updated confidence interval"
    
    def test_context_similarity(self, abductive_reasoning_engine):
        """
        Test context similarity computation
        
        Validates:
        - Context similarity calculation
        - Similarity score range
        """
        # Sample inputs
        observation = {
            'domain': 'technology',
            'description': 'Quantum computing advancement'
        }
        
        hypothesis_context = {
            'source': 'quantum_computing',
            'related_domains': ['technology', 'physics']
        }
        
        global_context = {
            'research_year': 2024,
            'global_trends': ['quantum technologies']
        }
        
        # Compute context similarity
        similarity = abductive_reasoning_engine._compute_context_similarity(
            observation, 
            hypothesis_context, 
            global_context
        )
        
        # Validate similarity score
        assert 0 <= similarity <= 1, "Invalid context similarity score"
    
    def test_domain_relevance(self, abductive_reasoning_engine):
        """
        Test domain relevance computation
        
        Validates:
        - Domain relevance calculation
        - Relevance score range
        """
        # Test cases with different domain alignments
        test_cases = [
            {
                'observation': {'domain': 'technology'},
                'hypothesis_context': {'source': 'quantum_computing'}
            },
            {
                'observation': {'domain': 'biology'},
                'hypothesis_context': {'source': 'machine_learning'}
            }
        ]
        
        for case in test_cases:
            relevance = abductive_reasoning_engine._compute_domain_relevance(
                case['observation'], 
                case['hypothesis_context']
            )
            
            # Validate relevance score
            assert 0 <= relevance <= 1, f"Invalid domain relevance for {case}"
