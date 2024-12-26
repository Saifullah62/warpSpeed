import asyncio
import logging
from typing import List, Dict, Any, Optional
from enum import Enum, auto

# NLP and ML Libraries
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# Image Processing
import cv2
import numpy as np
from PIL import Image

# Local imports
from .config import CONFIG
from .schema import Entity, EntityType
from .logging_config import get_logger, log_performance

class EntityExtractionStrategy(Enum):
    """
    Enumeration of entity extraction strategies.
    """
    TEXTUAL = auto()
    VISUAL = auto()
    CONTEXTUAL = auto()
    SEMANTIC = auto()

class MultiModalEntityExtractor:
    """
    Advanced multi-modal entity extraction system.
    
    Combines multiple techniques for comprehensive entity recognition:
    - Textual NER using SpaCy and Hugging Face models
    - Visual entity recognition from diagrams and images
    - Contextual and semantic understanding
    """
    
    def __init__(self, 
                 spacy_model: str = 'en_core_web_trf',
                 huggingface_model: str = 'allenai/scibert_scivocab_uncased'):
        """
        Initialize multi-modal entity extractor.
        
        Args:
            spacy_model: SpaCy model for NER
            huggingface_model: Hugging Face model for domain-specific NER
        """
        self.logger = get_logger(__name__)
        
        # Load SpaCy model
        try:
            self.spacy_nlp = spacy.load(spacy_model)
        except OSError:
            self.logger.warning(f"SpaCy model {spacy_model} not found. Downloading...")
            spacy.cli.download(spacy_model)
            self.spacy_nlp = spacy.load(spacy_model)
        
        # Load Hugging Face model
        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
        self.hf_model = AutoModelForTokenClassification.from_pretrained(huggingface_model)
        self.hf_ner_pipeline = pipeline(
            "ner", 
            model=self.hf_model, 
            tokenizer=self.tokenizer,
            aggregation_strategy='simple'
        )
        
        # Configuration
        self.config = CONFIG.get_config_section('entity_extraction')
    
    @log_performance()
    async def extract_entities(
        self, 
        text: Optional[str] = None, 
        image_path: Optional[str] = None,
        strategies: Optional[List[EntityExtractionStrategy]] = None
    ) -> List[Entity]:
        """
        Extract entities using multi-modal approaches.
        
        Args:
            text: Input text for textual entity extraction
            image_path: Path to image for visual entity extraction
            strategies: Specific extraction strategies to use
        
        Returns:
            List of extracted entities
        """
        # Default strategies if not provided
        if strategies is None:
            strategies = [
                EntityExtractionStrategy.TEXTUAL,
                EntityExtractionStrategy.CONTEXTUAL,
                EntityExtractionStrategy.SEMANTIC
            ]
        
        # Parallel entity extraction
        tasks = []
        
        if text and EntityExtractionStrategy.TEXTUAL in strategies:
            tasks.append(self._extract_textual_entities(text))
        
        if image_path and EntityExtractionStrategy.VISUAL in strategies:
            tasks.append(self._extract_visual_entities(image_path))
        
        # Combine results
        entity_results = await asyncio.gather(*tasks)
        
        # Flatten and deduplicate entities
        entities = [
            entity 
            for sublist in entity_results 
            for entity in sublist
        ]
        
        return self._reconcile_entities(entities)
    
    async def _extract_textual_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from textual content using multiple NER techniques.
        
        Args:
            text: Input text for entity extraction
        
        Returns:
            List of extracted textual entities
        """
        # SpaCy NER
        spacy_doc = self.spacy_nlp(text)
        spacy_entities = [
            Entity(
                name=ent.text, 
                type=self._map_spacy_type(ent.label_),
                properties={
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': 0.8  # SpaCy confidence
                }
            ) for ent in spacy_doc.ents
        ]
        
        # Hugging Face NER
        hf_entities = [
            Entity(
                name=ent['word'], 
                type=self._map_hf_type(ent['entity']),
                properties={
                    'start': ent['start'],
                    'end': ent['end'],
                    'confidence': ent['score']
                }
            ) for ent in self.hf_ner_pipeline(text)
        ]
        
        return spacy_entities + hf_entities
    
    async def _extract_visual_entities(self, image_path: str) -> List[Entity]:
        """
        Extract entities from scientific diagrams or images.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            List of extracted visual entities
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            
            # Preprocessing
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Text detection using Tesseract or similar OCR
            # Note: Requires additional OCR library integration
            
            # Object detection using pre-trained models
            # Note: Requires integration with object detection models
            
            # Placeholder for visual entity extraction
            return []
        
        except Exception as e:
            self.logger.error(f"Visual entity extraction failed: {e}")
            return []
    
    def _reconcile_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Reconcile and deduplicate extracted entities.
        
        Args:
            entities: List of extracted entities
        
        Returns:
            Reconciled list of unique entities
        """
        # Basic deduplication by name and type
        unique_entities = {}
        for entity in entities:
            key = (entity.name, entity.type)
            
            # Keep entity with highest confidence
            if key not in unique_entities or (
                entity.properties.get('confidence', 0) > 
                unique_entities[key].properties.get('confidence', 0)
            ):
                unique_entities[key] = entity
        
        return list(unique_entities.values())
    
    def _map_spacy_type(self, spacy_type: str) -> EntityType:
        """
        Map SpaCy entity types to our EntityType enum.
        
        Args:
            spacy_type: SpaCy entity label
        
        Returns:
            Mapped EntityType
        """
        type_mapping = {
            'ORG': EntityType.ORGANIZATION,
            'PERSON': EntityType.PERSON,
            'GPE': EntityType.LOCATION,
            'TECH': EntityType.TECHNOLOGY,
            'SCIENTIFIC_TERM': EntityType.CONCEPT
        }
        
        return type_mapping.get(spacy_type, EntityType.CONCEPT)
    
    def _map_hf_type(self, hf_type: str) -> EntityType:
        """
        Map Hugging Face entity types to our EntityType enum.
        
        Args:
            hf_type: Hugging Face entity label
        
        Returns:
            Mapped EntityType
        """
        type_mapping = {
            'B-ORG': EntityType.ORGANIZATION,
            'B-PER': EntityType.PERSON,
            'B-LOC': EntityType.LOCATION,
            'B-TECH': EntityType.TECHNOLOGY,
            'B-SCIENTIFIC_TERM': EntityType.CONCEPT
        }
        
        return type_mapping.get(hf_type, EntityType.CONCEPT)

# Optional: Async context manager for resource management
class MultiModalEntityExtractorManager:
    """
    Context manager for managing multi-modal entity extractor resources.
    """
    def __init__(self):
        self.extractor = None
    
    async def __aenter__(self):
        self.extractor = MultiModalEntityExtractor()
        return self.extractor
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Optional cleanup
        del self.extractor
