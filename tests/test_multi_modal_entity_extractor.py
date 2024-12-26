import os
import asyncio
import pytest
import numpy as np
from PIL import Image
import cv2

# Import the multi-modal entity extractor
from src.knowledge_graph.multi_modal_entity_extractor import (
    MultiModalEntityExtractor, 
    EntityExtractionStrategy,
    MultiModalEntityExtractorManager
)
from src.knowledge_graph.schema import EntityType

# Utility function to create test images
def create_test_image(filename, text=None):
    """
    Create a test image with optional text.
    
    Args:
        filename: Output filename for the test image
        text: Optional text to render on the image
    
    Returns:
        Path to the created image
    """
    # Ensure test image directory exists
    test_image_dir = os.path.join(os.path.dirname(__file__), 'test_images')
    os.makedirs(test_image_dir, exist_ok=True)
    
    # Create image
    img = Image.new('RGB', (400, 200), color='white')
    
    # Add text if provided
    if text:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype("arial.ttf", 20)
        draw.text((10, 10), text, fill='black', font=font)
    
    # Save image
    filepath = os.path.join(test_image_dir, filename)
    img.save(filepath)
    return filepath

@pytest.mark.asyncio
class TestMultiModalEntityExtractor:
    @pytest.fixture
    async def extractor(self):
        """
        Fixture to create a MultiModalEntityExtractor instance.
        """
        return MultiModalEntityExtractor()
    
    @pytest.mark.parametrize("text", [
        "Quantum entanglement is a fundamental principle of quantum mechanics.",
        "Warp drive technology could revolutionize space exploration.",
        "Gravitational wave detection using advanced laser interferometry has opened new frontiers in astrophysics."
    ])
    async def test_textual_entity_extraction(self, extractor, text):
        """
        Test textual entity extraction capabilities.
        
        Validates:
        - Successful entity extraction
        - Correct entity types
        - Non-empty entity names
        """
        # Extract entities
        entities = await extractor.extract_entities(
            text=text, 
            strategies=[EntityExtractionStrategy.TEXTUAL]
        )
        
        # Validate extraction
        assert len(entities) > 0, f"No entities extracted from text: {text}"
        
        # Check entity properties
        for entity in entities:
            assert entity.name, "Entity name cannot be empty"
            assert entity.type in EntityType, "Invalid entity type"
            assert len(entity.name) > 1, "Entity name too short"
            
            # Check confidence property
            assert 'confidence' in entity.properties, "Missing confidence score"
            assert 0 <= entity.properties['confidence'] <= 1, "Invalid confidence score"
    
    async def test_context_manager(self):
        """
        Test the async context manager for resource management.
        """
        async with MultiModalEntityExtractorManager() as extractor:
            # Test entity extraction within context
            text = "Quantum computing is an emerging field of computational science."
            entities = await extractor.extract_entities(text=text)
            
            assert len(entities) > 0, "Failed to extract entities in context manager"
    
    @pytest.mark.skipif(not os.path.exists("/usr/bin/tesseract"), reason="Tesseract OCR not installed")
    async def test_visual_entity_extraction(self, extractor):
        """
        Test visual entity extraction capabilities.
        
        Note: Requires Tesseract OCR to be installed.
        Validates:
        - Ability to process images
        - Extract text-based entities from images
        """
        # Create test image with scientific text
        test_image_path = create_test_image(
            'test_scientific_diagram.png', 
            text='Quantum Mechanics Principles'
        )
        
        # Extract visual entities
        entities = await extractor.extract_entities(
            image_path=test_image_path, 
            strategies=[EntityExtractionStrategy.VISUAL]
        )
        
        # Validate visual entity extraction
        assert len(entities) > 0, "No entities extracted from test image"
    
    async def test_multi_modal_extraction(self, extractor):
        """
        Test multi-modal entity extraction with combined strategies.
        
        Validates:
        - Successful extraction using multiple strategies
        - Entity deduplication
        """
        # Prepare test data
        text = "Quantum entanglement in superconducting quantum circuits"
        test_image_path = create_test_image(
            'quantum_circuit.png', 
            text='Superconducting Quantum Circuits'
        )
        
        # Extract entities using multiple strategies
        entities = await extractor.extract_entities(
            text=text, 
            image_path=test_image_path,
            strategies=[
                EntityExtractionStrategy.TEXTUAL,
                EntityExtractionStrategy.VISUAL,
                EntityExtractionStrategy.CONTEXTUAL
            ]
        )
        
        # Validate multi-modal extraction
        assert len(entities) > 0, "No entities extracted in multi-modal mode"
        
        # Check for unique entities
        entity_names = [entity.name for entity in entities]
        assert len(entity_names) == len(set(entity_names)), "Duplicate entities not properly reconciled"
    
    @pytest.mark.parametrize("strategies", [
        [EntityExtractionStrategy.TEXTUAL],
        [EntityExtractionStrategy.VISUAL],
        [EntityExtractionStrategy.CONTEXTUAL, EntityExtractionStrategy.SEMANTIC]
    ])
    async def test_extraction_strategies(self, extractor, strategies):
        """
        Test different entity extraction strategy combinations.
        
        Validates:
        - Ability to use different extraction strategies
        - Graceful handling of strategy combinations
        """
        # Prepare test data
        text = "Advanced quantum computing techniques in scientific research"
        
        # Extract entities with specific strategies
        entities = await extractor.extract_entities(
            text=text, 
            strategies=strategies
        )
        
        # Validate strategy-based extraction
        assert isinstance(entities, list), "Extraction should return a list of entities"
    
    def test_entity_type_mapping(self, extractor):
        """
        Test entity type mapping for various domain-specific terms.
        
        Validates:
        - Correct entity type assignment
        - Consistent type mapping
        """
        test_terms = [
            ("quantum mechanics", EntityType.CONCEPT),
            ("NASA research center", EntityType.ORGANIZATION),
            ("Albert Einstein", EntityType.PERSON),
            ("warp drive", EntityType.TECHNOLOGY)
        ]
        
        for term, expected_type in test_terms:
            # Use sync method for simplicity
            entities = asyncio.run(extractor.extract_entities(text=term))
            
            # Validate at least one entity is extracted
            assert len(entities) > 0, f"No entities extracted for term: {term}"
            
            # Check entity type
            entity_types = {entity.type for entity in entities}
            assert expected_type in entity_types, f"Incorrect type for term: {term}"

# Performance and Stress Testing
class TestMultiModalEntityExtractorPerformance:
    @pytest.mark.parametrize("num_texts", [10, 50, 100])
    @pytest.mark.asyncio
    async def test_large_scale_entity_extraction(self, num_texts):
        """
        Test entity extraction performance with large number of texts.
        
        Validates:
        - Ability to process multiple texts
        - Reasonable extraction time
        """
        extractor = MultiModalEntityExtractor()
        
        # Generate large text corpus
        sample_texts = [
            "Quantum mechanics is a fundamental theory in physics.",
            "Warp drive technology could enable faster-than-light travel.",
            "Gravitational wave detection reveals cosmic events."
        ]
        large_corpus = [text * (i+1) for i, text in enumerate(sample_texts * (num_texts // len(sample_texts) + 1))]
        large_corpus = large_corpus[:num_texts]
        
        # Measure extraction performance
        import time
        start_time = time.time()
        
        all_entities = []
        for text in large_corpus:
            entities = await extractor.extract_entities(text=text)
            all_entities.extend(entities)
        
        extraction_time = time.time() - start_time
        
        # Validate performance
        assert len(all_entities) > 0, "No entities extracted from large corpus"
        assert extraction_time < 10, f"Extraction took too long: {extraction_time} seconds"
