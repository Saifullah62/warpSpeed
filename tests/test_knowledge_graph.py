import asyncio
import pytest
import networkx as nx
from typing import List

# Import components to test
from src.knowledge_graph.advanced_entity_extractor import AdvancedEntityExtractor
from src.knowledge_graph.relationship_scoring import RelationshipConfidenceScorer
from src.knowledge_graph.relationship_mapper import RelationshipMapper
from src.knowledge_graph.builder import KnowledgeGraphBuilder
from src.knowledge_graph.schema import Entity, EntityType, Relationship, RelationshipType

# Sample texts for testing
SAMPLE_TEXTS = [
    "Quantum entanglement is a fundamental principle of quantum mechanics where two particles become correlated in such a way that the quantum state of each particle cannot be described independently.",
    
    "The development of warp drive technology could revolutionize space exploration by enabling faster-than-light travel through the manipulation of spacetime geometry.",
    
    "Gravitational wave detection using advanced laser interferometry has opened up new frontiers in observational astrophysics, allowing scientists to study cosmic events like black hole mergers."
]

@pytest.fixture
def advanced_entity_extractor():
    """Fixture for AdvancedEntityExtractor."""
    return AdvancedEntityExtractor()

@pytest.fixture
def relationship_confidence_scorer():
    """Fixture for RelationshipConfidenceScorer."""
    return RelationshipConfidenceScorer()

@pytest.mark.asyncio
class TestKnowledgeGraphComponents:
    async def test_advanced_entity_extraction(self, advanced_entity_extractor):
        """
        Test advanced entity extraction capabilities.
        
        Validates:
        - Correct number of entities extracted
        - Correct entity types
        - Non-empty entity names
        """
        for text in SAMPLE_TEXTS:
            entities = advanced_entity_extractor.extract_entities(text)
            
            # Validate extraction
            assert len(entities) > 0, f"No entities extracted from text: {text}"
            
            # Check entity properties
            for entity in entities:
                assert entity.name, "Entity name cannot be empty"
                assert entity.type in EntityType, "Invalid entity type"
                assert len(entity.name) > 1, "Entity name too short"
    
    def test_relationship_confidence_scoring(self, relationship_confidence_scorer):
        """
        Test relationship confidence scoring mechanism.
        
        Validates:
        - Confidence scores are between 0 and 1
        - Different relationship types have different base confidences
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
        
        # Create sample relationship
        relationship = Relationship(
            source_id=quantum_concept.id,
            target_id=warp_tech.id,
            type=RelationshipType.RELATES_TO
        )
        
        # Calculate confidence
        confidence = relationship_confidence_scorer.calculate_relationship_confidence(
            relationship, 
            quantum_concept, 
            warp_tech, 
            "Quantum mechanics principles could inform warp drive technology development"
        )
        
        # Validate confidence
        assert 0 <= confidence <= 1, "Confidence score must be between 0 and 1"
    
    @pytest.mark.parametrize("text", SAMPLE_TEXTS)
    async def test_relationship_mapping(self, text):
        """
        Test relationship mapping for a given text.
        
        Validates:
        - Successful relationship extraction
        - Reasonable number of relationships
        - Relationships have valid properties
        """
        # Extract entities
        extractor = AdvancedEntityExtractor()
        mapper = RelationshipMapper()
        
        # Extract entities
        entities = extractor.extract_entities(text)
        
        # Map relationships
        relationships = await mapper.map_relationships(entities, [text])
        
        # Validate relationships
        assert len(relationships) > 0, "No relationships extracted"
        
        for relationship in relationships:
            assert relationship.source_id is not None, "Relationship must have a source"
            assert relationship.target_id is not None, "Relationship must have a target"
            assert 0 <= relationship.confidence <= 1, "Relationship confidence must be between 0 and 1"
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_construction(self):
        """
        Test full knowledge graph construction process.
        
        Validates:
        - Successful graph creation
        - Graph contains nodes and edges
        - Graph metadata is correctly generated
        """
        # Prepare sample papers
        sample_papers = [
            {
                'title': 'Quantum Mechanics and Warp Drive',
                'abstract': ' '.join(SAMPLE_TEXTS),
                'content': ' '.join(SAMPLE_TEXTS)
            }
        ]
        
        # Initialize and build graph
        graph_builder = KnowledgeGraphBuilder()
        graph = await graph_builder.build_graph(sample_papers)
        
        # Validate graph
        assert graph is not None, "Graph construction failed"
        assert graph.number_of_nodes() > 0, "Graph must have nodes"
        assert graph.number_of_edges() > 0, "Graph must have edges"
    
    def test_entity_type_mapping(self, advanced_entity_extractor):
        """
        Test entity type mapping for various domain-specific terms.
        
        Validates:
        - Correct entity type assignment
        - Consistent type mapping
        """
        test_terms = [
            ("quantum mechanics", EntityType.CONCEPT),
            ("warp drive", EntityType.TECHNOLOGY),
            ("gravitational wave detection", EntityType.EXPERIMENT),
            ("laser interferometry", EntityType.TECHNOLOGY)
        ]
        
        for term, expected_type in test_terms:
            entities = advanced_entity_extractor.extract_entities(term)
            
            # Validate at least one entity is extracted
            assert len(entities) > 0, f"No entities extracted for term: {term}"
            
            # Check entity type
            entity_types = {entity.type for entity in entities}
            assert expected_type in entity_types, f"Incorrect type for term: {term}"

# Performance and Stress Testing
class TestKnowledgeGraphPerformance:
    @pytest.mark.parametrize("num_texts", [10, 50, 100])
    @pytest.mark.asyncio
    async def test_large_scale_entity_extraction(self, num_texts):
        """
        Test entity extraction performance with large number of texts.
        
        Validates:
        - Ability to process multiple texts
        - Reasonable extraction time
        """
        extractor = AdvancedEntityExtractor()
        
        # Generate large text corpus
        large_corpus = [text * (i+1) for i, text in enumerate(SAMPLE_TEXTS * (num_texts // len(SAMPLE_TEXTS) + 1))]
        large_corpus = large_corpus[:num_texts]
        
        # Measure extraction performance
        import time
        start_time = time.time()
        
        all_entities = []
        for text in large_corpus:
            entities = extractor.extract_entities(text)
            all_entities.extend(entities)
        
        extraction_time = time.time() - start_time
        
        # Validate performance
        assert len(all_entities) > 0, "No entities extracted from large corpus"
        assert extraction_time < 10, f"Extraction took too long: {extraction_time} seconds"
