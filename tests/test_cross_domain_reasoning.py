import asyncio
import pytest
import numpy as np

# Import cross-domain reasoning components
from src.knowledge_graph.cross_domain_reasoning import (
    CrossDomainReasoningSystem,
    LogicalInferenceEngine,
    ReasoningStrategy
)
from src.knowledge_graph.schema import Entity, EntityType, Relationship, RelationType

@pytest.fixture
def logical_inference_engine():
    """
    Fixture for creating a logical inference engine.
    """
    return LogicalInferenceEngine(domains=['physics', 'technology'])

@pytest.fixture
def cross_domain_reasoning_system():
    """
    Fixture for creating a cross-domain reasoning system.
    """
    return CrossDomainReasoningSystem(domains=['physics', 'technology'])

class TestLogicalInferenceEngine:
    def test_semantic_similarity_calculation(self, logical_inference_engine):
        """
        Test semantic similarity calculation between entities.
        
        Validates:
        - Successful similarity calculation
        - Similarity score properties
        """
        # Create sample entities
        quantum_concept = Entity(
            name="Quantum Mechanics", 
            type=EntityType.CONCEPT
        )
        warp_tech = Entity(
            name="Warp Drive Technology", 
            type=EntityType.TECHNOLOGY
        )
        
        # Calculate semantic similarity
        similarity = logical_inference_engine._calculate_semantic_similarity(
            quantum_concept, warp_tech
        )
        
        # Validate similarity
        assert isinstance(similarity, float), "Similarity should be a float"
        assert 0 <= similarity <= 1, "Similarity out of expected range"
    
    @pytest.mark.asyncio
    async def test_logical_inference_generation(self, logical_inference_engine):
        """
        Test generation of logical inferences across entities.
        
        Validates:
        - Successful inference generation
        - Inference structure and content
        """
        # Create sample entities from different domains
        entities = [
            Entity(name="Quantum Mechanics", type=EntityType.CONCEPT),
            Entity(name="Warp Drive", type=EntityType.TECHNOLOGY),
            Entity(name="Quantum Computing", type=EntityType.CONCEPT)
        ]
        
        # Create sample relationships
        relationships = [
            Relationship(
                source=entities[0], 
                target=entities[1], 
                type=RelationType.ENABLES
            )
        ]
        
        # Generate logical inferences
        inferences = await logical_inference_engine.generate_logical_inferences(
            entities, relationships
        )
        
        # Validate inferences
        assert isinstance(inferences, list), "Inferences should be a list"
        assert len(inferences) > 0, "No inferences generated"
        
        # Check inference structure
        for inference in inferences:
            assert 'entity1' in inference, "Missing entity1 in inference"
            assert 'entity2' in inference, "Missing entity2 in inference"
            assert 'similarity' in inference, "Missing similarity in inference"
            assert 'potential_relationships' in inference, "Missing potential relationships"
            assert 'inference_type' in inference, "Missing inference type"
    
    @pytest.mark.asyncio
    async def test_hypothesis_generation(self, logical_inference_engine):
        """
        Test generation of cross-domain hypotheses.
        
        Validates:
        - Successful hypothesis generation
        - Hypothesis structure and content
        """
        # Create sample entities from different domains
        entities = [
            Entity(name="Quantum Mechanics", type=EntityType.CONCEPT),
            Entity(name="Warp Drive", type=EntityType.TECHNOLOGY),
            Entity(name="Superconducting Circuit", type=EntityType.EXPERIMENT),
            Entity(name="Propulsion Research", type=EntityType.EXPERIMENT)
        ]
        
        # Generate hypotheses
        hypotheses = await logical_inference_engine.generate_hypotheses(entities)
        
        # Validate hypotheses
        assert isinstance(hypotheses, list), "Hypotheses should be a list"
        assert len(hypotheses) > 0, "No hypotheses generated"
        
        # Check hypothesis structure
        for hypothesis in hypotheses:
            assert 'type1' in hypothesis, "Missing type1 in hypothesis"
            assert 'type2' in hypothesis, "Missing type2 in hypothesis"
            assert 'avg_similarity' in hypothesis, "Missing average similarity"
            assert 'hypothesis' in hypothesis, "Missing hypothesis text"
            
            assert 0 <= hypothesis['avg_similarity'] <= 1, "Invalid similarity score"

class TestCrossDomainReasoningSystem:
    @pytest.mark.asyncio
    async def test_cross_domain_reasoning(self, cross_domain_reasoning_system):
        """
        Test comprehensive cross-domain reasoning.
        
        Validates:
        - Successful reasoning across domains
        - Reasoning results structure
        """
        # Create sample entities from different domains
        entities = [
            Entity(name="Quantum Mechanics", type=EntityType.CONCEPT),
            Entity(name="Warp Drive", type=EntityType.TECHNOLOGY),
            Entity(name="Quantum Computing", type=EntityType.CONCEPT),
            Entity(name="Propulsion Research", type=EntityType.EXPERIMENT)
        ]
        
        # Create sample relationships
        relationships = [
            Relationship(
                source=entities[0], 
                target=entities[1], 
                type=RelationType.ENABLES
            ),
            Relationship(
                source=entities[2], 
                target=entities[3], 
                type=RelationType.APPLIED_IN
            )
        ]
        
        # Perform cross-domain reasoning
        reasoning_results = await cross_domain_reasoning_system.reason_across_domains(
            entities, 
            relationships, 
            strategies=[
                ReasoningStrategy.LOGICAL_INFERENCE,
                ReasoningStrategy.HYPOTHESIS_GENERATION
            ]
        )
        
        # Validate reasoning results
        assert isinstance(reasoning_results, dict), "Reasoning results should be a dictionary"
        
        # Check logical inferences
        assert 'logical_inferences' in reasoning_results, "Missing logical inferences"
        assert isinstance(reasoning_results['logical_inferences'], list), "Logical inferences should be a list"
        
        # Check hypotheses
        assert 'hypotheses' in reasoning_results, "Missing hypotheses"
        assert isinstance(reasoning_results['hypotheses'], list), "Hypotheses should be a list"
    
    @pytest.mark.parametrize("num_entities", [10, 50, 100])
    @pytest.mark.asyncio
    async def test_large_scale_reasoning(self, cross_domain_reasoning_system, num_entities):
        """
        Test cross-domain reasoning performance with large number of entities.
        
        Validates:
        - Ability to process multiple entities
        - Reasonable reasoning time
        """
        # Generate large set of entities
        entities = [
            Entity(
                name=f"Entity_{i}", 
                type=EntityType(i % len(EntityType) + 1)
            ) for i in range(num_entities)
        ]
        
        # Generate relationships
        relationships = [
            Relationship(
                source=entities[i], 
                target=entities[(i+1) % num_entities], 
                type=RelationType.RELATES_TO
            ) for i in range(num_entities)
        ]
        
        # Measure reasoning performance
        import time
        start_time = time.time()
        
        reasoning_results = await cross_domain_reasoning_system.reason_across_domains(
            entities, relationships
        )
        
        reasoning_time = time.time() - start_time
        
        # Validate performance
        assert reasoning_results is not None, "No results returned from large-scale reasoning"
        assert reasoning_time < 15, f"Cross-domain reasoning took too long: {reasoning_time} seconds"
        
        # Check result structure
        assert 'logical_inferences' in reasoning_results, "Missing logical inferences"
        assert 'hypotheses' in reasoning_results, "Missing hypotheses"

# Advanced Reasoning Scenario Tests
class TestAdvancedReasoningScenarios:
    @pytest.mark.asyncio
    async def test_physics_technology_reasoning(self, cross_domain_reasoning_system):
        """
        Test reasoning between physics and technology domains.
        
        Validates:
        - Cross-domain inference quality
        - Meaningful relationship discovery
        """
        # Create physics and technology entities
        physics_entities = [
            Entity(name="Quantum Mechanics", type=EntityType.CONCEPT),
            Entity(name="Quantum Entanglement", type=EntityType.CONCEPT),
            Entity(name="Superconductivity", type=EntityType.CONCEPT)
        ]
        
        technology_entities = [
            Entity(name="Quantum Computing", type=EntityType.TECHNOLOGY),
            Entity(name="Superconducting Circuits", type=EntityType.TECHNOLOGY),
            Entity(name="Quantum Encryption", type=EntityType.TECHNOLOGY)
        ]
        
        # Combine entities
        all_entities = physics_entities + technology_entities
        
        # Create relationships
        relationships = [
            Relationship(
                source=physics_entities[0], 
                target=technology_entities[0], 
                type=RelationType.ENABLES
            )
        ]
        
        # Perform cross-domain reasoning
        reasoning_results = await cross_domain_reasoning_system.reason_across_domains(
            all_entities, relationships
        )
        
        # Validate advanced reasoning
        assert len(reasoning_results['logical_inferences']) > 0, "No cross-domain inferences generated"
        assert len(reasoning_results['hypotheses']) > 0, "No cross-domain hypotheses generated"
        
        # Check inference quality
        for inference in reasoning_results['logical_inferences']:
            assert inference['similarity'] > 0.5, "Weak cross-domain inference"
            assert len(inference['potential_relationships']) > 0, "No potential relationships found"
