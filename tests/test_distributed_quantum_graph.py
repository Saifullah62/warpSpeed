import pytest
import numpy as np
from qiskit import QuantumCircuit, Statevector
import networkx as nx
from typing import Dict, List, Any

from src.knowledge_graph.distributed_quantum_graph import (
    DistributedQuantumGraph,
    QuantumNode,
    DistributedShard
)

class TestDistributedQuantumGraph:
    @pytest.fixture
    def quantum_graph(self):
        """Create a quantum graph instance for testing"""
        return DistributedQuantumGraph(
            num_shards=4,
            quantum_simulation_depth=2
        )
    
    @pytest.fixture
    def sample_nodes(self) -> List[Dict[str, Any]]:
        """Create sample node data"""
        return [
            {
                'id': 'quantum_computing',
                'state_vector': np.random.rand(8) + 1j * np.random.rand(8),
                'properties': {
                    'domain': 'computer_science',
                    'confidence': 0.9
                }
            },
            {
                'id': 'quantum_entanglement',
                'state_vector': np.random.rand(8) + 1j * np.random.rand(8),
                'properties': {
                    'domain': 'physics',
                    'confidence': 0.85
                }
            }
        ]
    
    def test_node_creation(self, quantum_graph, sample_nodes):
        """Test node creation and properties"""
        # Add nodes
        nodes = []
        for node_data in sample_nodes:
            node = quantum_graph.add_node(
                node_id=node_data['id'],
                state_vector=node_data['state_vector'],
                properties=node_data['properties']
            )
            nodes.append(node)
        
        # Verify nodes
        assert len(nodes) == len(sample_nodes)
        for node, data in zip(nodes, sample_nodes):
            assert isinstance(node, QuantumNode)
            assert node.id == data['id']
            assert node.quantum_properties == data['properties']
            assert isinstance(node.superposition_weights, dict)
    
    def test_edge_creation(self, quantum_graph, sample_nodes):
        """Test edge creation and quantum entanglement"""
        # Add nodes
        for node_data in sample_nodes:
            quantum_graph.add_node(
                node_id=node_data['id'],
                state_vector=node_data['state_vector'],
                properties=node_data['properties']
            )
        
        # Add edge
        quantum_graph.add_edge(
            source_id=sample_nodes[0]['id'],
            target_id=sample_nodes[1]['id'],
            weight=0.8,
            entanglement_strength=0.6
        )
        
        # Verify edge in shards
        source_shard = quantum_graph._get_shard_for_node(sample_nodes[0]['id'])
        source_node = quantum_graph.shards[source_shard].nodes[sample_nodes[0]['id']]
        
        assert len(source_node.entanglement_pairs) > 0
        assert source_node.entanglement_pairs[0][0] == sample_nodes[1]['id']
    
    def test_subgraph_query(self, quantum_graph, sample_nodes):
        """Test quantum subgraph querying"""
        # Add nodes
        for node_data in sample_nodes:
            quantum_graph.add_node(
                node_id=node_data['id'],
                state_vector=node_data['state_vector'],
                properties=node_data['properties']
            )
        
        # Add edge
        quantum_graph.add_edge(
            source_id=sample_nodes[0]['id'],
            target_id=sample_nodes[1]['id'],
            weight=0.8,
            entanglement_strength=0.6
        )
        
        # Query subgraph
        node_ids = [node['id'] for node in sample_nodes]
        subgraph_nodes, adjacency_matrix = quantum_graph.query_subgraph(
            node_ids,
            quantum_enabled=True
        )
        
        # Verify results
        assert len(subgraph_nodes) == len(sample_nodes)
        assert isinstance(adjacency_matrix, np.ndarray)
        assert adjacency_matrix.shape == (len(sample_nodes), len(sample_nodes))
    
    def test_quantum_state_generation(self, quantum_graph):
        """Test quantum state generation"""
        state = quantum_graph._generate_quantum_state()
        
        # Verify state properties
        assert isinstance(state, np.ndarray)
        assert len(state) == 2 ** quantum_graph.quantum_config['num_qubits']
        assert np.isclose(np.linalg.norm(state), 1.0)
    
    def test_shard_management(self, quantum_graph, sample_nodes):
        """Test distributed shard management"""
        # Add nodes to different shards
        nodes = []
        for node_data in sample_nodes:
            node = quantum_graph.add_node(
                node_id=node_data['id'],
                state_vector=node_data['state_vector'],
                properties=node_data['properties']
            )
            nodes.append(node)
        
        # Verify shard distribution
        used_shards = set()
        for node in nodes:
            shard_id = quantum_graph._get_shard_for_node(node.id)
            used_shards.add(shard_id)
            assert node.id in quantum_graph.shards[shard_id].nodes
        
        # Verify shard quantum states
        for shard_id in used_shards:
            shard = quantum_graph.shards[shard_id]
            assert isinstance(shard.quantum_state, Statevector)
    
    def test_quantum_circuit_operations(self, quantum_graph, sample_nodes):
        """Test quantum circuit creation and execution"""
        # Add nodes
        for node_data in sample_nodes:
            quantum_graph.add_node(
                node_id=node_data['id'],
                state_vector=node_data['state_vector'],
                properties=node_data['properties']
            )
        
        # Create quantum circuit
        node_ids = [node['id'] for node in sample_nodes]
        subgraph_nodes, _ = quantum_graph.query_subgraph(node_ids)
        
        # Verify circuit execution
        for node_id, node in subgraph_nodes.items():
            assert isinstance(node.state_vector, np.ndarray)
            assert len(node.superposition_weights) > 0
    
    def test_entanglement_creation(self, quantum_graph, sample_nodes):
        """Test quantum entanglement creation"""
        # Add nodes
        for node_data in sample_nodes:
            quantum_graph.add_node(
                node_id=node_data['id'],
                state_vector=node_data['state_vector'],
                properties=node_data['properties']
            )
        
        # Create entanglement
        quantum_graph._create_entanglement(
            source_id=sample_nodes[0]['id'],
            target_id=sample_nodes[1]['id'],
            strength=0.7
        )
        
        # Verify entanglement pairs
        source_shard = quantum_graph._get_shard_for_node(sample_nodes[0]['id'])
        target_shard = quantum_graph._get_shard_for_node(sample_nodes[1]['id'])
        
        source_node = quantum_graph.shards[source_shard].nodes[sample_nodes[0]['id']]
        target_node = quantum_graph.shards[target_shard].nodes[sample_nodes[1]['id']]
        
        assert len(source_node.entanglement_pairs) == 1
        assert len(target_node.entanglement_pairs) == 1
        assert source_node.entanglement_pairs[0][0] == sample_nodes[1]['id']
        assert target_node.entanglement_pairs[0][0] == sample_nodes[0]['id']
    
    def test_quantum_state_update(self, quantum_graph, sample_nodes):
        """Test quantum state updates"""
        # Add node
        node = quantum_graph.add_node(
            node_id=sample_nodes[0]['id'],
            state_vector=sample_nodes[0]['state_vector'],
            properties=sample_nodes[0]['properties']
        )
        
        # Get shard
        shard_id = quantum_graph._get_shard_for_node(node.id)
        
        # Update shard quantum state
        quantum_graph._update_shard_quantum_state(shard_id)
        
        # Verify state update
        shard = quantum_graph.shards[shard_id]
        assert shard.quantum_state is not None
        assert isinstance(shard.quantum_state, Statevector)
