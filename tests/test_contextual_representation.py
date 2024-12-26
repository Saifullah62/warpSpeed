import asyncio
import pytest
import numpy as np

# Import contextual representation components
from src.knowledge_graph.contextual_representation import (
    ContextualRepresentationLearner,
    ContextualRepresentationConfig,
    DomainSpecificTokenizer
)
from src.knowledge_graph.schema import Entity, EntityType

@pytest.fixture
def contextual_representation_config():
    """
    Fixture for creating a contextual representation configuration.
    """
    return ContextualRepresentationConfig(domain='physics')

@pytest.fixture
def contextual_representation_learner(contextual_representation_config):
    """
    Fixture for creating a contextual representation learner.
    """
    return ContextualRepresentationLearner(contextual_representation_config)

@pytest.fixture
def domain_specific_tokenizer():
    """
    Fixture for creating a domain-specific tokenizer.
    """
    return DomainSpecificTokenizer(domain='physics')

class TestContextualRepresentationLearner:
    @pytest.mark.asyncio
    async def test_contextual_embedding_generation(self, contextual_representation_learner):
        """
        Test contextual embedding generation.
        
        Validates:
        - Successful embedding generation
        - Embedding vector properties
        """
        # Create sample entities
        entities = [
            Entity(name="Quantum Mechanics", type=EntityType.CONCEPT),
            Entity(name="Warp Drive", type=EntityType.TECHNOLOGY)
        ]
        
        # Provide context
        context = [
            "Advanced scientific research in quantum physics",
            "Theoretical propulsion technologies"
        ]
        
        # Generate contextual embeddings
        embeddings = await contextual_representation_learner.generate_contextual_embeddings(
            entities, context
        )
        
        # Validate embeddings
        assert len(embeddings) == len(entities), "Incorrect number of embeddings"
        
        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray), "Embedding should be a numpy array"
            assert embedding.size == 768, "Unexpected embedding size"
            assert not np.all(embedding == 0), "Embedding should not be zero vector"
    
    @pytest.mark.asyncio
    async def test_domain_representation_learning(self, contextual_representation_learner):
        """
        Test domain-specific representation learning.
        
        Validates:
        - Successful representation learning
        - Meaningful representation metrics
        """
        # Create sample entities
        entities = [
            Entity(name="Quantum Mechanics", type=EntityType.CONCEPT),
            Entity(name="Quantum Entanglement", type=EntityType.CONCEPT),
            Entity(name="Warp Drive", type=EntityType.TECHNOLOGY)
        ]
        
        # Provide training context
        training_context = [
            "Quantum mechanics explores fundamental principles of nature",
            "Quantum entanglement reveals deep interconnectedness of particles",
            "Theoretical propulsion technologies push boundaries of physics"
        ]
        
        # Learn domain representation
        representation_metrics = await contextual_representation_learner.learn_domain_representation(
            entities, training_context
        )
        
        # Validate representation metrics
        assert isinstance(representation_metrics, dict), "Representation metrics should be a dictionary"
        
        # Check specific metrics
        assert 'embedding_dimensions' in representation_metrics, "Missing embedding dimensions"
        assert 'unique_entities' in representation_metrics, "Missing unique entities count"
        assert 'embedding_variance' in representation_metrics, "Missing embedding variance"
        assert 'semantic_diversity' in representation_metrics, "Missing semantic diversity"
        
        # Validate metric ranges
        assert representation_metrics['embedding_dimensions'] == 768, "Incorrect embedding dimensions"
        assert representation_metrics['unique_entities'] > 0, "No unique entities found"
        assert 0 <= representation_metrics['embedding_variance'] <= 1, "Invalid embedding variance"
        assert 0 <= representation_metrics['semantic_diversity'] <= 1, "Invalid semantic diversity"

class TestDomainSpecificTokenizer:
    def test_tokenization(self, domain_specific_tokenizer):
        """
        Test domain-specific tokenization.
        
        Validates:
        - Successful tokenization
        - Domain-specific token handling
        """
        # Test scientific text
        text = "Quantum entanglement in superconducting circuits"
        tokens = domain_specific_tokenizer.tokenize(text)
        
        # Validate tokenization
        assert isinstance(tokens, list), "Tokenization failed"
        assert len(tokens) > 0, "No tokens generated"
        
        # Check for domain-specific tokens
        domain_specific_tokens = ['quantum', 'entanglement', 'superconducting']
        assert any(
            token.lower() in domain_specific_tokens 
            for token in tokens
        ), "Missing domain-specific tokens"

# Performance and Stress Testing
class TestContextualRepresentationPerformance:
    @pytest.mark.parametrize("num_entities", [10, 50, 100])
    @pytest.mark.asyncio
    async def test_large_scale_representation_learning(
        self, 
        contextual_representation_learner, 
        num_entities
    ):
        """
        Test contextual representation learning performance with large number of entities.
        
        Validates:
        - Ability to process multiple entities
        - Reasonable learning time
        """
        # Generate large set of entities
        entities = [
            Entity(
                name=f"Entity_{i}", 
                type=EntityType(i % len(EntityType) + 1)
            ) for i in range(num_entities)
        ]
        
        # Provide training context
        training_context = [
            f"Context for entity group {i}" for i in range(num_entities // 10 + 1)
        ]
        
        # Measure representation learning performance
        import time
        start_time = time.time()
        
        representation_metrics = await contextual_representation_learner.learn_domain_representation(
            entities, training_context
        )
        
        learning_time = time.time() - start_time
        
        # Validate performance
        assert representation_metrics is not None, "No metrics returned from large-scale representation learning"
        assert learning_time < 15, f"Representation learning took too long: {learning_time} seconds"

# Advanced Representation Learning Scenarios
class TestAdvancedRepresentationScenarios:
    @pytest.mark.asyncio
    async def test_multi_domain_representation(self, contextual_representation_learner):
        """
        Test representation learning across multiple domains.
        
        Validates:
        - Ability to handle diverse entity types
        - Meaningful representation across domains
        """
        # Create entities from different domains
        entities = [
            Entity(name="Quantum Mechanics", type=EntityType.CONCEPT),
            Entity(name="Warp Drive", type=EntityType.TECHNOLOGY),
            Entity(name="Superconducting Experiment", type=EntityType.EXPERIMENT),
            Entity(name="Propulsion Research", type=EntityType.EXPERIMENT)
        ]
        
        # Provide diverse context
        training_context = [
            "Quantum mechanics explores fundamental principles of physics",
            "Advanced propulsion technologies for space exploration",
            "Experimental research in superconductivity and energy systems"
        ]
        
        # Learn domain representation
        representation_metrics = await contextual_representation_learner.learn_domain_representation(
            entities, training_context
        )
        
        # Validate multi-domain representation
        assert representation_metrics['unique_entities'] == len(entities), "Not all entities represented"
        assert representation_metrics['semantic_diversity'] > 0.5, "Low semantic diversity across domains"
