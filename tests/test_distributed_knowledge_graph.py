import pytest
import numpy as np
import networkx as nx
import ray

from src.knowledge_graph.distributed_knowledge_graph import (
    DistributedKnowledgeGraphEngine,
    DistributedKnowledgeNode
)

@pytest.fixture
def sample_knowledge_graph():
    """Create a sample knowledge graph for testing"""
    G = nx.DiGraph()
    
    # Add sample nodes
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

class TestDistributedKnowledgeGraph:
    @pytest.fixture
    def distributed_graph_engine(self):
        """Create a distributed knowledge graph engine"""
        return DistributedKnowledgeGraphEngine(num_workers=4)
    
    def test_semantic_graph_partitioning(self, distributed_graph_engine, sample_knowledge_graph):
        """
        Test semantic graph partitioning
        
        Validates:
        - Correct number of partitions
        - Preservation of graph structure
        """
        partitions = distributed_graph_engine.semantic_graph_partitioning(sample_knowledge_graph)
        
        # Check partition count
        assert len(partitions) == 4, "Incorrect number of graph partitions"
        
        # Validate node distribution
        total_nodes = sum(len(partition.nodes()) for partition in partitions)
        assert total_nodes == len(sample_knowledge_graph.nodes()), "Node count mismatch"
    
    def test_quantum_state_computation(self, distributed_graph_engine):
        """
        Test quantum state computation for nodes
        
        Validates:
        - State vector generation
        - State normalization
        """
        test_node_data = {
            'id': 'test_node',
            'domain': 'test_domain',
            'description': 'Test node for quantum state computation'
        }
        
        # Compute quantum state
        quantum_state = ray.get(
            distributed_graph_engine._quantum_state_computation.remote(test_node_data)
        )
        
        # Validate quantum state properties
        assert isinstance(quantum_state, np.ndarray), "Invalid quantum state type"
        assert quantum_state.shape == (10,), "Incorrect quantum state dimension"
        
        # Check normalization
        norm = np.linalg.norm(quantum_state)
        assert np.isclose(norm, 1.0, atol=1e-7), "Quantum state not normalized"
    
    def test_quantum_entanglement_analysis(self, distributed_graph_engine, sample_knowledge_graph):
        """
        Test quantum entanglement analysis
        
        Validates:
        - Entanglement score computation
        - Score range and properties
        """
        # Add nodes to quantum graph
        for node, data in sample_knowledge_graph.nodes(data=True):
            distributed_graph_engine.add_quantum_node(node, data)
        
        # Compute entanglement scores
        entanglement_scores = distributed_graph_engine.quantum_entanglement_analysis()
        
        # Validate entanglement scores
        assert len(entanglement_scores) > 0, "No entanglement scores computed"
        
        for node, score in entanglement_scores.items():
            assert 0 <= score <= 1, f"Invalid entanglement score for {node}"
    
    def test_quantum_graph_inference(self, distributed_graph_engine, sample_knowledge_graph):
        """
        Test quantum-inspired graph inference
        
        Validates:
        - Inference result generation
        - Relevance score computation
        """
        # Add nodes to quantum graph
        for node, data in sample_knowledge_graph.nodes(data=True):
            distributed_graph_engine.add_quantum_node(node, data)
        
        # Perform inference from a query node
        query_node = 'quantum_computing'
        inferred_nodes = distributed_graph_engine.quantum_graph_inference(query_node)
        
        # Validate inference results
        assert len(inferred_nodes) > 0, "No nodes inferred"
        
        # Check relevance scores
        for node, score in inferred_nodes:
            assert 0 <= score <= 1, f"Invalid relevance score for {node}"
        
        # Verify sorted order (descending relevance)
        relevance_scores = [score for _, score in inferred_nodes]
        assert relevance_scores == sorted(relevance_scores, reverse=True), "Relevance scores not sorted"

# Cleanup after tests
def test_cleanup():
    """Ensure distributed resources are properly shutdown"""
    from src.knowledge_graph.distributed_knowledge_graph import cleanup_distributed_resources
    cleanup_distributed_resources()
