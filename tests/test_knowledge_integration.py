import pytest
import networkx as nx
import torch

from src.knowledge_graph.knowledge_integration import (
    KnowledgeGraphInterface,
    ReasoningEngine,
    ResearchDirectionGenerator,
    KnowledgeSource,
    KnowledgeSourceType,
    initialize_knowledge_integration_system
)
from src.knowledge_graph.schema import Entity, EntityType
from src.knowledge_graph.advanced_embedding import MultiModalEmbeddingFinetuner

class TestKnowledgeGraphInterface:
    @pytest.fixture
    def embedding_model(self):
        """Create a multi-modal embedding fine-tuner."""
        return MultiModalEmbeddingFinetuner()
    
    @pytest.fixture
    def knowledge_graph(self, embedding_model):
        """Create a knowledge graph interface."""
        return KnowledgeGraphInterface(embedding_model)
    
    def test_entity_addition(self, knowledge_graph):
        """
        Test adding entities to the knowledge graph.
        
        Validates:
        - Successful entity addition
        - Embedding generation
        - Graph size management
        """
        # Create test entities
        entities = [
            Entity(name="Quantum Computer", type=EntityType.TECHNOLOGY),
            Entity(name="Neural Network", type=EntityType.CONCEPT),
            Entity(name="Machine Learning", type=EntityType.CONCEPT)
        ]
        
        # Add knowledge source
        source = KnowledgeSource(
            source_type=KnowledgeSourceType.ACADEMIC_PAPER,
            identifier="test_source",
            reliability_score=0.9
        )
        
        # Add entities
        for entity in entities:
            result = knowledge_graph.add_entity(entity, source)
            assert result, f"Failed to add entity: {entity.name}"
        
        # Validate graph properties
        assert len(knowledge_graph.graph.nodes) == len(entities), "Incorrect number of nodes"
        
        # Check node attributes
        for entity in entities:
            assert knowledge_graph.graph.nodes[entity.name]['type'] == entity.type
            assert 'embedding' in knowledge_graph.graph.nodes[entity.name]
    
    def test_relationship_addition(self, knowledge_graph):
        """
        Test adding relationships between entities.
        
        Validates:
        - Relationship creation
        - Edge attribute tracking
        """
        # Create test entities
        entities = [
            Entity(name="Quantum Computing", type=EntityType.TECHNOLOGY),
            Entity(name="Quantum Mechanics", type=EntityType.CONCEPT)
        ]
        
        # Add entities first
        for entity in entities:
            knowledge_graph.add_entity(entity)
        
        # Add relationship
        result = knowledge_graph.add_relationship(
            entities[0], 
            entities[1], 
            relationship_type="DERIVED_FROM"
        )
        
        assert result, "Failed to add relationship"
        
        # Validate graph relationship
        assert knowledge_graph.graph.has_edge(
            entities[0].name, 
            entities[1].name
        ), "Relationship not added to graph"
        
        edge_data = knowledge_graph.graph.edges[
            entities[0].name, 
            entities[1].name
        ]
        assert edge_data['type'] == "DERIVED_FROM", "Incorrect relationship type"
    
    def test_graph_pruning(self, embedding_model):
        """
        Test knowledge graph pruning mechanism.
        
        Validates:
        - Capacity management
        - Least connected entity removal
        """
        # Create knowledge graph with small capacity
        knowledge_graph = KnowledgeGraphInterface(
            embedding_model, 
            max_entities=10
        )
        
        # Add more entities than capacity
        for i in range(20):
            entity = Entity(
                name=f"Entity_{i}", 
                type=EntityType(i % len(EntityType) + 1)
            )
            knowledge_graph.add_entity(entity)
        
        # Validate pruning
        assert len(knowledge_graph.graph.nodes) <= 10, "Graph not pruned correctly"

class TestReasoningEngine:
    @pytest.fixture
    def knowledge_integration_system(self):
        """
        Create a complete knowledge integration system.
        """
        return initialize_knowledge_integration_system()
    
    def test_multi_hop_reasoning_initialization(self, knowledge_integration_system):
        """
        Test initialization and basic functionality of reasoning engine.
        
        Validates:
        - Successful system initialization
        - Reasoning engine setup
        """
        knowledge_graph, reasoning_engine, _ = knowledge_integration_system
        
        assert reasoning_engine is not None, "Reasoning engine not initialized"
        assert reasoning_engine.knowledge_graph == knowledge_graph, "Incorrect knowledge graph reference"
    
    def test_reasoning_query_processing(self, knowledge_integration_system):
        """
        Test reasoning query processing.
        
        Validates:
        - Query processing mechanism
        - Reasoning path generation
        """
        _, reasoning_engine, _ = knowledge_integration_system
        
        # Test reasoning query
        query = "What is the relationship between quantum computing and machine learning?"
        
        reasoning_result = reasoning_engine.multi_hop_reasoning(query)
        
        # Validate reasoning result structure
        assert isinstance(reasoning_result, dict), "Invalid reasoning result type"
        assert 'entities' in reasoning_result, "Missing entities in reasoning result"
        assert 'relationships' in reasoning_result, "Missing relationships in reasoning result"
        assert 'confidence' in reasoning_result, "Missing confidence in reasoning result"
        assert 'explanation' in reasoning_result, "Missing explanation in reasoning result"

class TestResearchDirectionGenerator:
    @pytest.fixture
    def knowledge_integration_system(self):
        """
        Create a complete knowledge integration system.
        """
        return initialize_knowledge_integration_system()
    
    def test_research_gap_identification(self, knowledge_integration_system):
        """
        Test research gap identification mechanism.
        
        Validates:
        - Gap identification process
        - Priority scoring
        """
        _, _, research_generator = knowledge_integration_system
        
        # Test research gap identification
        domain = "Quantum Computing"
        existing_research = [
            "Quantum Error Correction",
            "Quantum Algorithms"
        ]
        
        research_gaps = research_generator.identify_research_gaps(
            domain, 
            existing_research
        )
        
        # Validate research gaps
        assert isinstance(research_gaps, dict), "Invalid research gaps type"
        assert len(research_gaps) > 0, "No research gaps identified"
        
        # Check gap priority scores
        for gap, priority in research_gaps.items():
            assert 0 <= priority <= 1, f"Invalid priority score for gap: {gap}"
    
    def test_research_roadmap_generation(self, knowledge_integration_system):
        """
        Test research roadmap generation.
        
        Validates:
        - Roadmap creation
        - Time horizon management
        """
        _, _, research_generator = knowledge_integration_system
        
        # Generate research roadmap
        initial_focus = "Quantum Machine Learning"
        time_horizon = 5
        
        roadmap = research_generator.generate_research_roadmap(
            initial_focus, 
            time_horizon
        )
        
        # Validate roadmap structure
        assert isinstance(roadmap, dict), "Invalid roadmap type"
        assert roadmap['initial_focus'] == initial_focus, "Incorrect initial focus"
        assert roadmap['time_horizon'] == time_horizon, "Incorrect time horizon"
        assert 'research_stages' in roadmap, "Missing research stages"

def test_knowledge_integration_system_initialization():
    """
    Test complete knowledge integration system initialization.
    
    Validates:
    - System components creation
    - Correct component types
    """
    knowledge_graph, reasoning_engine, research_generator = initialize_knowledge_integration_system()
    
    # Validate system components
    assert knowledge_graph is not None, "Knowledge graph not initialized"
    assert reasoning_engine is not None, "Reasoning engine not initialized"
    assert research_generator is not None, "Research direction generator not initialized"
    
    # Check component types
    from src.knowledge_graph.knowledge_integration import (
        KnowledgeGraphInterface,
        ReasoningEngine,
        ResearchDirectionGenerator
    )
    
    assert isinstance(knowledge_graph, KnowledgeGraphInterface), "Invalid knowledge graph type"
    assert isinstance(reasoning_engine, ReasoningEngine), "Invalid reasoning engine type"
    assert isinstance(research_generator, ResearchDirectionGenerator), "Invalid research generator type"
