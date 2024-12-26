import spacy
import numpy as np
from typing import List, Dict, Set, Tuple, Optional
import logging
from .schema import Entity, EntityType, Property
import hashlib
import re
import asyncio

logger = logging.getLogger(__name__)

class EntityExtractor:
    def __init__(self):
        # Load SpaCy model for scientific text processing
        self.nlp = spacy.load("en_core_sci_lg")
        
        # Physics-specific entity patterns
        self.physics_patterns = [
            {"label": "CONCEPT", "pattern": [{"LOWER": {"IN": ["quantum", "relativistic", "gravitational", "electromagnetic"]}}]},
            {"label": "THEORY", "pattern": [{"LOWER": {"IN": ["theory", "principle", "law", "equation"]}}]},
            {"label": "PHENOMENON", "pattern": [{"LOWER": {"IN": ["effect", "interaction", "force", "field"]}}]},
            {"label": "TECHNOLOGY", "pattern": [{"LOWER": {"IN": ["drive", "engine", "generator", "detector"]}}]},
        ]
        
        # Add patterns to the pipeline
        ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(self.physics_patterns)
        
    async def extract_entities(self, text: str) -> List[Entity]:
        """Extract physics-related entities from text using async method."""
        try:
            # Use run_in_executor to make the synchronous SpaCy processing async
            loop = asyncio.get_running_loop()
            doc = await loop.run_in_executor(None, self.nlp, text)
            
            entities = []
            
            # Process named entities
            for ent in doc.ents:
                if self._is_relevant_entity(ent.text):
                    entity_type = self._determine_entity_type(ent)
                    if entity_type:
                        entity = self._create_entity(ent.text, entity_type, doc)
                        entities.append(entity)
            
            # Extract equations
            equations = await loop.run_in_executor(None, self._extract_equations, text)
            for eq in equations:
                entity = self._create_equation_entity(eq)
                entities.append(entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []

    def _is_relevant_entity(self, text: str) -> bool:
        """Determine if an entity is relevant for knowledge graph."""
        # Filter out very short or common words
        if len(text) < 2:
            return False
        
        # List of common stop words and irrelevant terms to filter out
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
        
        # Convert to lowercase and check
        text_lower = text.lower().strip()
        return text_lower not in stop_words and len(text_lower) > 1

    def _determine_entity_type(self, ent) -> Optional[EntityType]:
        """Determine the most appropriate entity type based on SpaCy entity."""
        # Map SpaCy labels to our EntityType
        label_mapping = {
            'ORG': EntityType.TECHNOLOGY,
            'PERSON': EntityType.CONCEPT,
            'GPE': EntityType.CONCEPT,
            'LOC': EntityType.CONCEPT,
            'PRODUCT': EntityType.TECHNOLOGY,
            'EVENT': EntityType.PHENOMENON
        }
        
        # Check custom physics patterns first
        for pattern in self.physics_patterns:
            if pattern['label'].lower() in ent.label_.lower():
                return EntityType(pattern['label'].lower())
        
        # Fallback to SpaCy label mapping
        return label_mapping.get(ent.label_, EntityType.CONCEPT)

    def _create_entity(self, text: str, entity_type: EntityType, doc) -> Entity:
        """Create an Entity object with extracted information."""
        # Generate a unique ID
        entity_id = hashlib.md5(text.encode()).hexdigest()
        
        # Extract context and properties
        context = self._extract_context(text, doc)
        
        return Entity(
            id=entity_id,
            name=text,
            type=entity_type,
            description=context.get('description', ''),
            confidence=0.8,  # Default confidence
            properties=[
                Property(
                    name='context', 
                    value=context.get('context', ''), 
                    confidence=0.7
                )
            ]
        )

    def _extract_context(self, text: str, doc) -> Dict[str, str]:
        """Extract contextual information for an entity."""
        context = {}
        
        # Find sentences containing the entity
        for sent in doc.sents:
            if text in sent.text:
                context['context'] = sent.text
                context['description'] = self._generate_description(sent)
                break
        
        return context

    def _generate_description(self, sent) -> str:
        """Generate a concise description from a sentence."""
        # Simple description generation
        description = sent.text
        
        # Truncate if too long
        if len(description) > 250:
            description = description[:250] + '...'
        
        return description

    def _extract_equations(self, text: str) -> List[str]:
        """Extract mathematical equations from text."""
        # Regular expression to find LaTeX-style or basic mathematical equations
        equation_patterns = [
            r'\$[^\$]+\$',  # LaTeX inline equations
            r'\\\([^\)]+\\\)',  # LaTeX display equations
            r'[A-Z]+\s*=\s*[A-Z0-9]+',  # Basic equation pattern
        ]
        
        equations = []
        for pattern in equation_patterns:
            equations.extend(re.findall(pattern, text))
        
        return equations

    def _create_equation_entity(self, equation: str) -> Entity:
        """Create an entity specifically for mathematical equations."""
        entity_id = hashlib.md5(equation.encode()).hexdigest()
        
        return Entity(
            id=entity_id,
            name=equation,
            type=EntityType.EQUATION,
            description=f"Mathematical equation: {equation}",
            confidence=0.9,
            properties=[
                Property(
                    name='equation_type', 
                    value='symbolic', 
                    confidence=0.8
                )
            ]
        )
