import asyncio
import pytest
import numpy as np

# Import the entity disambiguation components
from src.knowledge_graph.entity_disambiguation import (
    EntityDisambiguator,
    DisambiguationStrategy,
    DomainKnowledgeBase
)
from src.knowledge_graph.schema import Entity, EntityType

@pytest.fixture
def domain_knowledge_base():
    """
    Fixture for creating a domain knowledge base.
    """
    return DomainKnowledgeBase(domain='physics')

@pytest.fixture
def entity_disambiguator():
    """
    Fixture for creating an entity disambiguator.
    """
    return EntityDisambiguator(domain='physics')

class TestDomainKnowledgeBase:
    def test_semantic_similarity_initialization(self, domain_knowledge_base):
        """
        Test initialization of domain knowledge base.
        
        Validates:
        - Successful embedding model loading
        - Embedding model configuration
        """
        assert domain_knowledge_base.embedding_model is not None, "Embedding model not loaded"
        assert 'tokenizer' in domain_knowledge_base.embedding_model, "Missing tokenizer"
        assert 'model' in domain_knowledge_base.embedding_model, "Missing model"
    
    def test_semantic_similarity_calculation(self, domain_knowledge_base):
        """
        Test semantic similarity calculation.
        
        Validates:
        - Similarity calculation between entities
        - Similarity score range
        """
        # Create sample entities
        quantum_entity1 = Entity(
            name="Quantum Mechanics", 
            type=EntityType.CONCEPT
        )
        quantum_entity2 = Entity(
            name="Quantum Physics", 
            type=EntityType.CONCEPT
        )
        
        # Different type entities
        different_entity = Entity(
            name="Warp Drive", 
            type=EntityType.TECHNOLOGY
        )
        
        # Calculate similarities
        similar_score = domain_knowledge_base.calculate_semantic_similarity(
            quantum_entity1, quantum_entity2
        )
        different_score = domain_knowledge_base.calculate_semantic_similarity(
            quantum_entity1, different_entity
        )
        
        # Validate similarity scores
        assert 0 <= similar_score <= 1, "Similarity score out of range"
        assert 0 <= different_score <= 1, "Similarity score out of range"
        
        # Similar entities should have higher similarity
        assert similar_score >= different_score, "Semantic similarity calculation incorrect"

@pytest.mark.asyncio
class TestEntityDisambiguator:
    async def test_semantic_disambiguation(self, entity_disambiguator):
        """
        Test semantic disambiguation strategy.
        
        Validates:
        - Successful entity disambiguation
        - Correct handling of multiple entities
        """
        # Create ambiguous entities
        entities = [
            Entity(name="Quantum Mechanics", type=EntityType.CONCEPT),
            Entity(name="Quantum Physics", type=EntityType.CONCEPT),
            Entity(name="Quantum Computing", type=EntityType.CONCEPT)
        ]
        
        # Disambiguate entities
        disambiguated_entities = await entity_disambiguator.disambiguate_entities(
            entities, 
            strategies=[DisambiguationStrategy.SEMANTIC]
        )
        
        # Validate disambiguation
        assert len(disambiguated_entities) > 0, "No entities returned after disambiguation"
        
        # Check semantic similarity properties
        for entity in disambiguated_entities:
            assert 'semantic_similarity' in entity.properties, "Missing semantic similarity"
            assert 0 <= entity.properties['semantic_similarity'] <= 1, "Invalid similarity score"
    
    async def test_multi_strategy_disambiguation(self, entity_disambiguator):
        """
        Test disambiguation using multiple strategies.
        
        Validates:
        - Successful multi-strategy disambiguation
        - Consistent entity reconciliation
        """
        # Create diverse entities
        entities = [
            Entity(name="Quantum Mechanics", type=EntityType.CONCEPT),
            Entity(name="Quantum Physics", type=EntityType.CONCEPT),
            Entity(name="Warp Drive", type=EntityType.TECHNOLOGY),
            Entity(name="Space Propulsion", type=EntityType.TECHNOLOGY)
        ]
        
        # Disambiguate using multiple strategies
        disambiguated_entities = await entity_disambiguator.disambiguate_entities(
            entities, 
            strategies=[
                DisambiguationStrategy.SEMANTIC,
                DisambiguationStrategy.CONTEXTUAL,
                DisambiguationStrategy.CROSS_REFERENCE
            ]
        )
        
        # Validate disambiguation
        assert len(disambiguated_entities) > 0, "No entities returned after disambiguation"
        
        # Ensure entities are grouped by type
        concept_entities = [e for e in disambiguated_entities if e.type == EntityType.CONCEPT]
        tech_entities = [e for e in disambiguated_entities if e.type == EntityType.TECHNOLOGY]
        
        assert len(concept_entities) > 0, "No concept entities found"
        assert len(tech_entities) > 0, "No technology entities found"
    
    def test_entity_type_preservation(self, entity_disambiguator):
        """
        Test that entity types are preserved during disambiguation.
        
        Validates:
        - Entity type consistency
        - No type transformation during disambiguation
        """
        # Create entities with different types
        entities = [
            Entity(name="Quantum Mechanics", type=EntityType.CONCEPT),
            Entity(name="NASA", type=EntityType.ORGANIZATION),
            Entity(name="Laser Interferometry", type=EntityType.EXPERIMENT)
        ]
        
        # Disambiguate entities
        disambiguated_entities = asyncio.run(
            entity_disambiguator.disambiguate_entities(entities)
        )
        
        # Validate type preservation
        for original, disambiguated in zip(entities, disambiguated_entities):
            assert original.type == disambiguated.type, "Entity type changed during disambiguation"

# Performance and Stress Testing
class TestEntityDisambiguationPerformance:
    @pytest.mark.parametrize("num_entities", [10, 50, 100])
    @pytest.mark.asyncio
    async def test_large_scale_disambiguation(self, entity_disambiguator, num_entities):
        """
        Test entity disambiguation performance with large number of entities.
        
        Validates:
        - Ability to process multiple entities
        - Reasonable disambiguation time
        """
        # Generate large set of entities
        entities = [
            Entity(
                name=f"Entity_{i}", 
                type=EntityType(i % len(EntityType) + 1)
            ) for i in range(num_entities)
        ]
        
        # Measure disambiguation performance
        import time
        start_time = time.time()
        
        disambiguated_entities = await entity_disambiguator.disambiguate_entities(entities)
        
        disambiguation_time = time.time() - start_time
        
        # Validate performance
        assert len(disambiguated_entities) > 0, "No entities returned from large-scale disambiguation"
        assert disambiguation_time < 10, f"Disambiguation took too long: {disambiguation_time} seconds"
