import asyncio
import pytest
import numpy as np

# Import domain evaluation components
from src.knowledge_graph.domain_evaluation import (
    DomainSpecificTokenizer,
    DomainEvaluator,
    DomainEvaluationMetrics
)
from src.knowledge_graph.schema import Entity, EntityType, Relationship, RelationshipType

@pytest.fixture
def domain_specific_tokenizer():
    """
    Fixture for creating a domain-specific tokenizer.
    """
    return DomainSpecificTokenizer(domain='physics')

@pytest.fixture
def domain_evaluator():
    """
    Fixture for creating a domain evaluator.
    """
    return DomainEvaluator(domain='physics')

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
    
    def test_contextual_embeddings(self, domain_specific_tokenizer):
        """
        Test contextual embedding generation.
        
        Validates:
        - Successful embedding generation
        - Embedding vector properties
        """
        # Test tokens
        tokens = ['quantum', 'mechanics', 'theory']
        
        # Generate embeddings
        embeddings = domain_specific_tokenizer.generate_contextual_embeddings(tokens)
        
        # Validate embeddings
        assert isinstance(embeddings, np.ndarray), "Embedding should be a numpy array"
        assert embeddings.size == 768, "Unexpected embedding size"
        assert not np.all(embeddings == 0), "Embedding should not be zero vector"

class TestDomainEvaluator:
    @pytest.mark.asyncio
    async def test_domain_understanding_evaluation(self, domain_evaluator):
        """
        Test comprehensive domain understanding evaluation.
        
        Validates:
        - Successful metrics generation
        - Reasonable metric values
        """
        # Create sample entities
        entities = [
            Entity(name="Quantum Mechanics", type=EntityType.CONCEPT),
            Entity(name="Warp Drive", type=EntityType.TECHNOLOGY),
            Entity(name="Quantum Computing Experiment", type=EntityType.EXPERIMENT)
        ]
        
        # Create sample relationships
        relationships = [
            Relationship(
                source=entities[0], 
                target=entities[1], 
                type=RelationshipType.ENABLES
            ),
            Relationship(
                source=entities[1], 
                target=entities[2], 
                type=RelationshipType.APPLIED_IN
            )
        ]
        
        # Evaluate domain understanding
        metrics = await domain_evaluator.evaluate_domain_understanding(
            entities, relationships
        )
        
        # Validate metrics
        assert isinstance(metrics, DomainEvaluationMetrics), "Invalid metrics type"
        
        # Check individual metric ranges
        assert 0 <= metrics.entity_recognition_precision <= 1, "Invalid precision"
        assert 0 <= metrics.entity_recognition_recall <= 1, "Invalid recall"
        assert 0 <= metrics.entity_recognition_f1 <= 1, "Invalid F1 score"
        
        assert 0 <= metrics.relationship_inference_accuracy <= 1, "Invalid relationship accuracy"
        
        assert 0 <= metrics.semantic_similarity_score <= 1, "Invalid semantic similarity"
        assert 0 <= metrics.contextual_coherence_score <= 1, "Invalid contextual coherence"
        assert 0 <= metrics.domain_adaptation_score <= 1, "Invalid domain adaptation score"
    
    def test_entity_recognition_metrics(self, domain_evaluator):
        """
        Test entity recognition metric calculations.
        
        Validates:
        - Precision calculation
        - Recall calculation
        - F1 score calculation
        """
        # Create sample entities
        entities = [
            Entity(name="Quantum Mechanics", type=EntityType.CONCEPT),
            Entity(name="Warp Drive", type=EntityType.TECHNOLOGY),
            Entity(name="Random Entity", type=EntityType.ORGANIZATION)
        ]
        
        # Calculate precision
        precision = domain_evaluator._calculate_entity_recognition_precision(entities)
        recall = domain_evaluator._calculate_entity_recognition_recall(entities)
        f1_score = domain_evaluator._calculate_f1_score(precision, recall)
        
        # Validate metrics
        assert 0 <= precision <= 1, "Invalid precision"
        assert 0 <= recall <= 1, "Invalid recall"
        assert 0 <= f1_score <= 1, "Invalid F1 score"
    
    def test_semantic_similarity(self, domain_evaluator):
        """
        Test semantic similarity calculation.
        
        Validates:
        - Similarity calculation between entities
        - Similarity score range
        """
        # Create sample entities
        entities = [
            Entity(name="Quantum Mechanics", type=EntityType.CONCEPT),
            Entity(name="Quantum Physics", type=EntityType.CONCEPT),
            Entity(name="Warp Drive", type=EntityType.TECHNOLOGY)
        ]
        
        # Calculate semantic similarity
        similarity_score = domain_evaluator._calculate_semantic_similarity(entities)
        
        # Validate similarity score
        assert 0 <= similarity_score <= 1, "Invalid semantic similarity score"
        
        # Similar entities should have higher similarity
        assert similarity_score > 0, "Semantic similarity should be positive"

# Performance and Stress Testing
class TestDomainEvaluationPerformance:
    @pytest.mark.parametrize("num_entities", [10, 50, 100])
    @pytest.mark.asyncio
    async def test_large_scale_domain_evaluation(self, domain_evaluator, num_entities):
        """
        Test domain evaluation performance with large number of entities.
        
        Validates:
        - Ability to process multiple entities
        - Reasonable evaluation time
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
                type=RelationshipType.RELATES_TO
            ) for i in range(num_entities)
        ]
        
        # Measure evaluation performance
        import time
        start_time = time.time()
        
        metrics = await domain_evaluator.evaluate_domain_understanding(
            entities, relationships
        )
        
        evaluation_time = time.time() - start_time
        
        # Validate performance
        assert metrics is not None, "No metrics returned from large-scale evaluation"
        assert evaluation_time < 10, f"Domain evaluation took too long: {evaluation_time} seconds"
