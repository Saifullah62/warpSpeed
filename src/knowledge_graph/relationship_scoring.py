import math
import logging
from typing import List, Dict, Any, Optional
from .schema import Relationship, Entity

logger = logging.getLogger(__name__)

class RelationshipConfidenceScorer:
    """
    Advanced confidence scoring system for knowledge graph relationships.
    
    Calculates confidence based on multiple factors:
    - Textual co-occurrence
    - Semantic proximity
    - Entity type compatibility
    - Contextual relevance
    """
    
    def __init__(self, 
                 context_weight: float = 0.3,
                 type_compatibility_weight: float = 0.2,
                 co_occurrence_weight: float = 0.3,
                 semantic_proximity_weight: float = 0.2):
        """
        Initialize confidence scorer with configurable weights.
        
        Args:
            context_weight: Weight for contextual relevance
            type_compatibility_weight: Weight for entity type compatibility
            co_occurrence_weight: Weight for textual co-occurrence
            semantic_proximity_weight: Weight for semantic proximity
        """
        self.context_weight = context_weight
        self.type_compatibility_weight = type_compatibility_weight
        self.co_occurrence_weight = co_occurrence_weight
        self.semantic_proximity_weight = semantic_proximity_weight
    
    def _calculate_type_compatibility(self, source: Entity, target: Entity) -> float:
        """
        Calculate compatibility score between two entities based on their types.
        
        Args:
            source: Source entity
            target: Target entity
        
        Returns:
            Compatibility score between 0 and 1
        """
        type_compatibility_matrix = {
            # Define type compatibility rules
            ('concept', 'technology'): 0.7,
            ('technology', 'concept'): 0.7,
            ('theory', 'concept'): 0.8,
            ('concept', 'theory'): 0.8,
            ('experiment', 'technology'): 0.6,
            ('technology', 'experiment'): 0.6
        }
        
        key = (source.type.value.lower(), target.type.value.lower())
        return type_compatibility_matrix.get(key, 0.5)
    
    def _calculate_co_occurrence(self, source: Entity, target: Entity, context: str) -> float:
        """
        Calculate co-occurrence score based on entity names in context.
        
        Args:
            source: Source entity
            target: Target entity
            context: Text context to analyze
        
        Returns:
            Co-occurrence score between 0 and 1
        """
        context_lower = context.lower()
        source_name_lower = source.name.lower()
        target_name_lower = target.name.lower()
        
        # Check proximity of entity names in context
        source_index = context_lower.find(source_name_lower)
        target_index = context_lower.find(target_name_lower)
        
        if source_index == -1 or target_index == -1:
            return 0.0
        
        # Calculate proximity score
        proximity_score = 1.0 / (1 + abs(source_index - target_index) / len(context))
        return proximity_score
    
    def _calculate_semantic_proximity(self, source: Entity, target: Entity) -> float:
        """
        Calculate semantic proximity between entities.
        
        Note: This is a placeholder for more advanced semantic analysis.
        In a full implementation, this would use embeddings or semantic networks.
        
        Args:
            source: Source entity
            target: Target entity
        
        Returns:
            Semantic proximity score between 0 and 1
        """
        # Basic implementation using name similarity
        name_similarity = self._calculate_name_similarity(source.name, target.name)
        return name_similarity
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate name similarity using Levenshtein distance.
        
        Args:
            name1: First name
            name2: Second name
        
        Returns:
            Similarity score between 0 and 1
        """
        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)
            
            if len(s2) == 0:
                return len(s1)
            
            previous_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
        
        # Normalize Levenshtein distance
        max_len = max(len(name1), len(name2))
        distance = levenshtein_distance(name1.lower(), name2.lower())
        similarity = 1.0 - (distance / max_len)
        return max(0.0, min(1.0, similarity))
    
    def calculate_relationship_confidence(
        self, 
        relationship: Relationship, 
        source: Entity, 
        target: Entity, 
        context: Optional[str] = None
    ) -> float:
        """
        Calculate comprehensive confidence score for a relationship.
        
        Args:
            relationship: Relationship to score
            source: Source entity
            target: Target entity
            context: Optional text context for co-occurrence analysis
        
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Calculate individual component scores
            type_compatibility = self._calculate_type_compatibility(source, target)
            semantic_proximity = self._calculate_semantic_proximity(source, target)
            
            # Co-occurrence score (optional)
            co_occurrence_score = 0.5  # Default neutral score
            if context:
                co_occurrence_score = self._calculate_co_occurrence(source, target, context)
            
            # Combine scores with weighted average
            confidence = (
                (self.type_compatibility_weight * type_compatibility) +
                (self.semantic_proximity_weight * semantic_proximity) +
                (self.co_occurrence_weight * co_occurrence_score)
            )
            
            # Normalize to ensure score is between 0 and 1
            confidence = max(0.0, min(1.0, confidence))
            
            # Add base relationship type confidence
            base_confidence = {
                'RELATES_TO': 0.5,
                'DEPENDS_ON': 0.7,
                'CONTRADICTS': 0.3
            }.get(relationship.type.value, 0.5)
            
            # Final confidence is a weighted combination
            final_confidence = (confidence * 0.7) + (base_confidence * 0.3)
            
            return final_confidence
        
        except Exception as e:
            logger.warning(f"Error calculating relationship confidence: {e}")
            return 0.5  # Default neutral confidence
    
    def refine_graph_relationships(
        self, 
        relationships: List[Relationship], 
        entities: List[Entity], 
        context: Optional[str] = None
    ) -> List[Relationship]:
        """
        Refine and filter relationships based on confidence scoring.
        
        Args:
            relationships: List of initial relationships
            entities: List of entities
            context: Optional text context for analysis
        
        Returns:
            Filtered and scored relationships
        """
        # Create entity lookup for efficient access
        entity_lookup = {entity.id: entity for entity in entities}
        
        refined_relationships = []
        for relationship in relationships:
            try:
                source = entity_lookup.get(relationship.source_id)
                target = entity_lookup.get(relationship.target_id)
                
                if not source or not target:
                    continue
                
                # Calculate confidence
                confidence = self.calculate_relationship_confidence(
                    relationship, source, target, context
                )
                
                # Update relationship confidence
                relationship.confidence = confidence
                
                # Optional: Filter out low-confidence relationships
                if confidence > 0.3:
                    refined_relationships.append(relationship)
            
            except Exception as e:
                logger.warning(f"Error processing relationship: {e}")
        
        return refined_relationships
