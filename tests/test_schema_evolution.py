import asyncio
import pytest
import networkx as nx
import numpy as np

# Import schema evolution components
from src.knowledge_graph.schema_evolution import (
    DynamicSchemaManager,
    RelationshipInferenceModel,
    SchemaEvolutionStrategy
)
from src.knowledge_graph.schema import Entity, EntityType, RelationshipType

@pytest.fixture
def relationship_inference_model():
    """
    Fixture for creating a relationship inference model.
    """
    return RelationshipInferenceModel(domain='physics')

@pytest.fixture
def dynamic_schema_manager():
    """
    Fixture for creating a dynamic schema manager.
    """
    return DynamicSchemaManager(domain='physics')

class TestRelationshipInferenceModel:
    def test_embedding_calculation(self, relationship_inference_model):
        """
        Test relationship embedding calculation.
        
        Validates:
        - Successful embedding generation
        - Embedding vector properties
        """
        # Create sample entities
        quantum_concept = Entity(
            name="Quantum Mechanics", 
            type=EntityType.CONCEPT
        )
        warp_tech = Entity(
            name="Warp Drive", 
            type=EntityType.TECHNOLOGY
        )
        
        # Calculate embedding
        embedding = relationship_inference_model.calculate_relationship_embedding(
            quantum_concept, warp_tech, "Theoretical physics research"
        )
        
        # Validate embedding
        assert isinstance(embedding, np.ndarray), "Embedding should be a numpy array"
        assert embedding.size == 768, "Unexpected embedding size"
        assert not np.all(embedding == 0), "Embedding should not be zero vector"
    
    def test_relationship_type_inference(self, relationship_inference_model):
        """
        Test relationship type inference.
        
        Validates:
        - Successful relationship type prediction
        - Reasonable type selection
        """
        # Create sample entities
        quantum_concept = Entity(
            name="Quantum Mechanics", 
            type=EntityType.CONCEPT
        )
        warp_tech = Entity(
            name="Warp Drive", 
            type=EntityType.TECHNOLOGY
        )
        
        # Infer relationship type
        relationship_type = relationship_inference_model.infer_relationship_type(
            quantum_concept, warp_tech, "Theoretical foundations"
        )
        
        # Validate relationship type
        assert relationship_type in RelationshipType, "Invalid relationship type"
        assert relationship_type == RelationshipType.ENABLES, "Unexpected relationship type"

@pytest.mark.asyncio
class TestDynamicSchemaManager:
    async def test_incremental_schema_update(self, dynamic_schema_manager):
        """
        Test incremental schema update strategy.
        
        Validates:
        - Successful graph update
        - Correct node addition
        """
        # Create initial graph
        initial_graph = nx.DiGraph()
        initial_graph.add_node(
            'quantum_concept', 
            name='Quantum Mechanics', 
            type=EntityType.CONCEPT
        )
        
        # New entities to add
        new_entities = [
            Entity(
                name="Warp Drive", 
                type=EntityType.TECHNOLOGY
            ),
            Entity(
                name="Quantum Computing", 
                type=EntityType.CONCEPT
            )
        ]
        
        # Evolve schema
        evolved_graph = await dynamic_schema_manager.evolve_schema(
            initial_graph, 
            new_entities, 
            strategies=[SchemaEvolutionStrategy.INCREMENTAL_UPDATE]
        )
        
        # Validate schema evolution
        assert len(evolved_graph.nodes) == 3, "Incorrect number of nodes after update"
        
        # Check new nodes
        for entity in new_entities:
            assert any(
                evolved_graph.nodes[node]['name'] == entity.name 
                for node in evolved_graph.nodes
            ), f"Entity {entity.name} not added to graph"
    
    async def test_semantic_schema_inference(self, dynamic_schema_manager):
        """
        Test semantic schema inference strategy.
        
        Validates:
        - Relationship inference
        - Correct edge addition
        """
        # Create initial graph
        initial_graph = nx.DiGraph()
        initial_graph.add_node(
            'quantum_concept', 
            name='Quantum Mechanics', 
            type=EntityType.CONCEPT
        )
        
        # New entities to add
        new_entities = [
            Entity(
                name="Warp Drive", 
                type=EntityType.TECHNOLOGY
            )
        ]
        
        # Evolve schema
        evolved_graph = await dynamic_schema_manager.evolve_schema(
            initial_graph, 
            new_entities, 
            strategies=[SchemaEvolutionStrategy.SEMANTIC_INFERENCE]
        )
        
        # Validate schema evolution
        assert len(evolved_graph.edges) > 0, "No relationships inferred"
        
        # Check inferred relationships
        for u, v, data in evolved_graph.edges(data=True):
            assert 'type' in data, "Missing relationship type"
            assert data['type'] in [rt.value for rt in RelationshipType], "Invalid relationship type"
    
    async def test_multi_strategy_schema_evolution(self, dynamic_schema_manager):
        """
        Test schema evolution using multiple strategies.
        
        Validates:
        - Successful multi-strategy evolution
        - Consistent graph updates
        """
        # Create initial graph
        initial_graph = nx.DiGraph()
        initial_graph.add_node(
            'quantum_concept', 
            name='Quantum Mechanics', 
            type=EntityType.CONCEPT
        )
        
        # New entities to add
        new_entities = [
            Entity(
                name="Warp Drive", 
                type=EntityType.TECHNOLOGY
            ),
            Entity(
                name="Quantum Computing", 
                type=EntityType.CONCEPT
            )
        ]
        
        # Evolve schema using multiple strategies
        evolved_graph = await dynamic_schema_manager.evolve_schema(
            initial_graph, 
            new_entities, 
            strategies=[
                SchemaEvolutionStrategy.INCREMENTAL_UPDATE,
                SchemaEvolutionStrategy.SEMANTIC_INFERENCE
            ]
        )
        
        # Validate schema evolution
        assert len(evolved_graph.nodes) > 1, "Insufficient nodes after evolution"
        assert len(evolved_graph.edges) > 0, "No relationships inferred"
    
    def test_schema_versioning(self, dynamic_schema_manager):
        """
        Test schema versioning capabilities.
        
        Validates:
        - Successful version saving
        - Version limit enforcement
        """
        # Create sample graph
        graph = nx.DiGraph()
        graph.add_node(
            'quantum_concept', 
            name='Quantum Mechanics', 
            type=EntityType.CONCEPT
        )
        
        # Save multiple versions
        for _ in range(15):  # Exceed default max versions
            dynamic_schema_manager.save_schema_version(graph)
        
        # Check version tracking
        assert len(dynamic_schema_manager.schema_versions) <= 10, "Exceeded maximum versions"

# Performance and Stress Testing
class TestSchemaEvolutionPerformance:
    @pytest.mark.parametrize("num_entities", [10, 50, 100])
    @pytest.mark.asyncio
    async def test_large_scale_schema_evolution(self, dynamic_schema_manager, num_entities):
        """
        Test schema evolution performance with large number of entities.
        
        Validates:
        - Ability to process multiple entities
        - Reasonable evolution time
        """
        # Create initial graph
        initial_graph = nx.DiGraph()
        initial_graph.add_node(
            'initial_concept', 
            name='Initial Concept', 
            type=EntityType.CONCEPT
        )
        
        # Generate large set of entities
        new_entities = [
            Entity(
                name=f"Entity_{i}", 
                type=EntityType(i % len(EntityType) + 1)
            ) for i in range(num_entities)
        ]
        
        # Measure schema evolution performance
        import time
        start_time = time.time()
        
        evolved_graph = await dynamic_schema_manager.evolve_schema(
            initial_graph, 
            new_entities
        )
        
        evolution_time = time.time() - start_time
        
        # Validate performance
        assert len(evolved_graph.nodes) > 1, "No nodes added during large-scale evolution"
        assert evolution_time < 10, f"Schema evolution took too long: {evolution_time} seconds"
