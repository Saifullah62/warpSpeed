import spacy
import re
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from .schema import Entity, EntityType
from .entity_extractor import EntityExtractor

logger = logging.getLogger(__name__)

class AdvancedEntityExtractor(EntityExtractor):
    """
    Advanced entity extraction system with multi-modal and domain-specific capabilities.
    
    Features:
    - Hybrid NER using multiple models
    - Domain-specific entity recognition
    - Contextual entity disambiguation
    - Multi-modal entity extraction
    """
    
    def __init__(
        self, 
        spacy_model: str = 'en_core_web_trf', 
        huggingface_model: str = 'allenai/scibert_scivocab_uncased'
    ):
        """
        Initialize advanced entity extractor.
        
        Args:
            spacy_model: SpaCy model for initial NER
            huggingface_model: Hugging Face model for scientific domain NER
        """
        super().__init__()
        
        # Load SpaCy model
        try:
            self.nlp = spacy.load(spacy_model)
        except Exception as e:
            logger.warning(f"Could not load SpaCy model {spacy_model}: {e}")
            self.nlp = spacy.load('en_core_web_sm')
        
        # Load Hugging Face scientific NER model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(huggingface_model)
            self.ner_model = AutoModelForTokenClassification.from_pretrained(huggingface_model)
            self.ner_pipeline = pipeline(
                "ner", 
                model=self.ner_model, 
                tokenizer=self.tokenizer
            )
        except Exception as e:
            logger.warning(f"Could not load Hugging Face model {huggingface_model}: {e}")
            self.ner_pipeline = None
    
    def extract_entities(
        self, 
        text: str, 
        additional_context: Optional[Dict[str, Any]] = None
    ) -> List[Entity]:
        """
        Extract entities using advanced multi-modal techniques.
        
        Args:
            text: Input text for entity extraction
            additional_context: Optional context for entity disambiguation
        
        Returns:
            List of extracted entities
        """
        # Combine multiple NER techniques
        entities = []
        
        # 1. SpaCy NER
        spacy_entities = self._extract_spacy_entities(text)
        entities.extend(spacy_entities)
        
        # 2. Scientific Domain NER (Hugging Face)
        if self.ner_pipeline:
            scientific_entities = self._extract_scientific_entities(text)
            entities.extend(scientific_entities)
        
        # 3. Custom domain-specific entity extraction
        custom_entities = self._extract_domain_specific_entities(text)
        entities.extend(custom_entities)
        
        # 4. Contextual disambiguation and filtering
        entities = self._disambiguate_entities(entities, additional_context)
        
        # Remove duplicates while preserving order
        unique_entities = []
        seen = set()
        for entity in entities:
            if entity.name.lower() not in seen:
                unique_entities.append(entity)
                seen.add(entity.name.lower())
        
        return unique_entities
    
    def _extract_spacy_entities(self, text: str) -> List[Entity]:
        """
        Extract entities using SpaCy NER.
        
        Args:
            text: Input text
        
        Returns:
            List of entities extracted by SpaCy
        """
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            try:
                entity_type = self._map_spacy_entity_type(ent.label_)
                entity = Entity(
                    name=ent.text,
                    type=entity_type,
                    properties={
                        'spacy_label': ent.label_,
                        'start_char': ent.start_char,
                        'end_char': ent.end_char
                    }
                )
                entities.append(entity)
            except Exception as e:
                logger.warning(f"Error processing SpaCy entity: {e}")
        
        return entities
    
    def _extract_scientific_entities(self, text: str) -> List[Entity]:
        """
        Extract scientific domain entities using Hugging Face NER.
        
        Args:
            text: Input text
        
        Returns:
            List of scientific entities
        """
        if not self.ner_pipeline:
            return []
        
        try:
            ner_results = self.ner_pipeline(text)
            
            entities = []
            current_entity = None
            
            for result in ner_results:
                if result['entity'].startswith('B-'):
                    # Beginning of a new entity
                    if current_entity:
                        entities.append(current_entity)
                    
                    entity_type = result['entity'][2:]
                    current_entity = Entity(
                        name=result['word'],
                        type=self._map_scientific_entity_type(entity_type),
                        properties={'scientific_label': entity_type}
                    )
                elif result['entity'].startswith('I-') and current_entity:
                    # Continue current entity
                    current_entity.name += f" {result['word']}"
            
            # Add last entity
            if current_entity:
                entities.append(current_entity)
            
            return entities
        
        except Exception as e:
            logger.warning(f"Error in scientific entity extraction: {e}")
            return []
    
    def _extract_domain_specific_entities(self, text: str) -> List[Entity]:
        """
        Extract domain-specific entities using custom rules.
        
        Args:
            text: Input text
        
        Returns:
            List of domain-specific entities
        """
        entities = []
        
        # Physics and technology-specific patterns
        patterns = [
            # Physics concepts
            (r'\b(quantum|relativistic|gravitational|electromagnetic)\s+\w+', EntityType.CONCEPT),
            
            # Technology patterns
            (r'\b(warp|propulsion|energy|drive|reactor)\s+\w+', EntityType.TECHNOLOGY),
            
            # Experimental techniques
            (r'\b(experiment|measurement|observation|simulation)\s+\w+', EntityType.EXPERIMENT)
        ]
        
        for pattern, entity_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(Entity(
                    name=match.group(0),
                    type=entity_type,
                    properties={'extraction_method': 'domain_specific_regex'}
                ))
        
        return entities
    
    def _disambiguate_entities(
        self, 
        entities: List[Entity], 
        context: Optional[Dict[str, Any]] = None
    ) -> List[Entity]:
        """
        Disambiguate and refine extracted entities.
        
        Args:
            entities: List of extracted entities
            context: Optional context for disambiguation
        
        Returns:
            Refined list of entities
        """
        # Sort entities by length (longer, more specific entities first)
        entities.sort(key=lambda x: len(x.name), reverse=True)
        
        # Remove overlapping or redundant entities
        refined_entities = []
        for entity in entities:
            # Check if entity is already covered by a more specific entity
            if not any(
                entity.name.lower() in existing.name.lower() and 
                existing.name.lower() != entity.name.lower()
                for existing in refined_entities
            ):
                refined_entities.append(entity)
        
        return refined_entities
    
    def _map_spacy_entity_type(self, spacy_type: str) -> EntityType:
        """
        Map SpaCy entity types to our custom entity types.
        
        Args:
            spacy_type: SpaCy NER label
        
        Returns:
            Mapped EntityType
        """
        mapping = {
            'ORG': EntityType.TECHNOLOGY,
            'PERSON': EntityType.CONCEPT,
            'GPE': EntityType.CONCEPT,
            'LOC': EntityType.CONCEPT,
            'PRODUCT': EntityType.TECHNOLOGY,
            'EVENT': EntityType.EXPERIMENT
        }
        return mapping.get(spacy_type, EntityType.CONCEPT)
    
    def _map_scientific_entity_type(self, scientific_type: str) -> EntityType:
        """
        Map scientific NER types to our custom entity types.
        
        Args:
            scientific_type: Scientific NER label
        
        Returns:
            Mapped EntityType
        """
        mapping = {
            'METHOD': EntityType.EXPERIMENT,
            'MATERIAL': EntityType.TECHNOLOGY,
            'METRIC': EntityType.CONCEPT,
            'TASK': EntityType.EXPERIMENT
        }
        return mapping.get(scientific_type, EntityType.CONCEPT)
