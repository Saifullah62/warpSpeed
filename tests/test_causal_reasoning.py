import pytest
import networkx as nx
import numpy as np

from src.reasoning.causal_reasoning_engine import (
    CausalReasoningEngine,
    CausalRelationship
)

@pytest.fixture
def sample_knowledge_graph():
    """Create a sample knowledge graph for causal reasoning"""
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

class TestCausalReasoningEngine:
    @pytest.fixture
    def causal_reasoning_engine(self, sample_knowledge_graph):
        """Create a causal reasoning engine with sample knowledge graph"""
        return CausalReasoningEngine(sample_knowledge_graph)
    
    def test_causal_relationship_learning(self, causal_reasoning_engine):
        """
        Test learning causal relationships
        
        Validates:
        - Causal relationship creation
        - Strength and confidence computation
        """
        source = 'quantum_computing'
        target = 'quantum_algorithms'
        
        # Learn causal relationship
        causal_rel = causal_reasoning_engine.learn_causal_relationship(
            source, 
            target, 
            data={'temporal_lag': 0.5}
        )
        
        # Validate causal relationship properties
        assert isinstance(causal_rel, CausalRelationship), "Invalid causal relationship type"
        
        # Check relationship attributes
        assert causal_rel.source == source, "Incorrect source node"
        assert causal_rel.target == target, "Incorrect target node"
        
        # Validate strength and confidence
        assert 0 <= causal_rel.strength <= 1, "Invalid causal strength"
        assert 0 <= causal_rel.confidence <= 1, "Invalid confidence score"
        assert 0 <= causal_rel.intervention_effect <= 1, "Invalid intervention effect"
    
    def test_causal_graph_structure(self, causal_reasoning_engine):
        """
        Test causal graph structure and properties
        
        Validates:
        - Graph initialization
        - Node and edge preservation
        """
        causal_graph = causal_reasoning_engine.causal_graph
        
        # Check graph properties
        assert len(causal_graph.nodes()) > 0, "Empty causal graph"
        assert len(causal_graph.edges()) > 0, "No causal relationships"
        
        # Verify node attributes
        for node in causal_graph.nodes():
            assert 'domain' in causal_graph.nodes[node], f"Missing domain for node {node}"
    
    def test_causal_intervention(self, causal_reasoning_engine):
        """
        Test causal intervention analysis
        
        Validates:
        - Intervention effect computation
        - Downstream impact tracking
        """
        intervention_node = 'quantum_computing'
        
        # Perform causal intervention
        intervention_effects = causal_reasoning_engine.perform_causal_intervention(
            intervention_node, 
            intervention_type='direct'
        )
        
        # Validate intervention effects
        assert isinstance(intervention_effects, dict), "Invalid intervention effects type"
        
        # Check intervention history
        assert len(causal_reasoning_engine.intervention_history) > 0, "No intervention recorded"
        
        # Validate intervention effect properties
        for node, effect in intervention_effects.items():
            assert 'intervention_type' in effect, f"Missing intervention type for {node}"
            assert 'propagation_effect' in effect, f"Missing propagation effect for {node}"
            assert 'affected_relationships' in effect, f"Missing affected relationships for {node}"
            
            assert 0 <= effect['propagation_effect'] <= 1, f"Invalid propagation effect for {node}"
    
    def test_causal_path_analysis(self, causal_reasoning_engine):
        """
        Test causal path analysis
        
        Validates:
        - Path discovery
        - Relationship strength computation
        """
        source = 'quantum_computing'
        target = 'machine_learning'
        
        # Analyze causal paths
        causal_paths = causal_reasoning_engine.analyze_causal_paths(
            source, 
            target, 
            max_path_length=3
        )
        
        # Validate causal paths
        assert isinstance(causal_paths, list), "Invalid causal paths type"
        
        # Check path properties
        for path_info in causal_paths:
            assert 'path' in path_info, "Missing path in path analysis"
            assert 'relationships' in path_info, "Missing relationships in path analysis"
            assert 'total_strength' in path_info, "Missing total strength in path analysis"
            
            # Verify path structure
            assert path_info['path'][0] == source, "Incorrect source node in path"
            assert path_info['path'][-1] == target, "Incorrect target node in path"
            
            # Check relationship properties
            for rel in path_info['relationships']:
                assert 'source' in rel, "Missing source in relationship"
                assert 'target' in rel, "Missing target in relationship"
                assert 'strength' in rel, "Missing strength in relationship"
                assert 'confidence' in rel, "Missing confidence in relationship"
                
                assert 0 <= rel['strength'] <= 1, "Invalid relationship strength"
                assert 0 <= rel['confidence'] <= 1, "Invalid relationship confidence"
    
    def test_domain_proximity(self, causal_reasoning_engine):
        """
        Test domain proximity computation
        
        Validates:
        - Domain proximity calculation
        - Proximity score range
        """
        test_cases = [
            ('quantum_computing', 'quantum_algorithms'),  # Same domain
            ('quantum_computing', 'machine_learning')     # Different domains
        ]
        
        for source, target in test_cases:
            proximity = causal_reasoning_engine._compute_domain_proximity(source, target)
            
            # Validate proximity score
            assert 0 <= proximity <= 1, f"Invalid proximity between {source} and {target}"
            
            # Check domain-based proximity logic
            if source == target or causal_reasoning_engine.causal_graph.nodes[source].get('domain') == \
                    causal_reasoning_engine.causal_graph.nodes[target].get('domain'):
                assert proximity >= 0.5, f"Unexpected proximity for similar domains: {source}, {target}"
            else:
                assert proximity < 1.0, f"Unexpected proximity for different domains: {source}, {target}"
