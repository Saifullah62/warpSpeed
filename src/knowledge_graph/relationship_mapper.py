import spacy
from typing import List, Dict, Tuple, Set, Optional
import networkx as nx
from .schema import EntityType, Entity, Relationship, RelationType
import hashlib
import logging
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import asyncio
from .relationship_scoring import RelationshipConfidenceScorer

logger = logging.getLogger(__name__)

class RelationshipMapper:
    def __init__(self):
        self.nlp = spacy.load("en_core_sci_lg")
        self.graph = nx.DiGraph()
        
    async def map_relationships(self, entities: List[Entity], texts: List[str]) -> List[Relationship]:
        """Map relationships between entities based on text analysis."""
        try:
            # Use run_in_executor to make synchronous methods async
            loop = asyncio.get_running_loop()
            
            # Create entity embeddings
            entity_embeddings = await loop.run_in_executor(
                None, self._create_entity_embeddings, entities
            )
            
            # Find semantic relationships
            semantic_rels = await loop.run_in_executor(
                None, self._find_semantic_relationships, entities, entity_embeddings
            )
            
            # Find dependency relationships from texts
            dependency_rels = await loop.run_in_executor(
                None, self._find_dependency_relationships, entities, texts
            )
            
            # Find citation-based relationships
            citation_rels = await loop.run_in_executor(
                None, self._find_citation_relationships, entities
            )
            
            # Combine and validate relationships
            relationships = semantic_rels + dependency_rels + citation_rels
            valid_relationships = await loop.run_in_executor(
                None, self._validate_relationships, relationships
            )
            
            return valid_relationships
            
        except Exception as e:
            logger.error(f"Error mapping relationships: {str(e)}")
            return []
            
    def _create_entity_embeddings(self, entities: List[Entity]) -> Dict[str, np.ndarray]:
        """Create embeddings for entities using SpaCy."""
        embeddings = {}
        embedding_list = []
        entity_ids = []
        
        for entity in entities:
            # Combine name and description for better embedding
            text = f"{entity.name} {entity.description}"
            doc = self.nlp(text)
            
            # Use document vector as embedding
            if len(doc.vector) > 0:
                embeddings[entity.id] = doc.vector
                embedding_list.append(doc.vector)
                entity_ids.append(entity.id)
        
        # Convert to numpy array for cosine similarity
        if embedding_list:
            embedding_matrix = np.array(embedding_list)
            
            # Normalize embeddings
            embedding_matrix = embedding_matrix / np.linalg.norm(embedding_matrix, axis=1)[:, np.newaxis]
            
            # Store normalized embeddings
            for idx, entity_id in enumerate(entity_ids):
                embeddings[entity_id] = embedding_matrix[idx]
        
        return embeddings

    def _find_semantic_relationships(self, 
                                     entities: List[Entity], 
                                     embeddings: Dict[str, np.ndarray]) -> List[Relationship]:
        """Find semantic relationships between entities based on embedding similarity."""
        relationships = []
        
        # Ensure we have embeddings and convert to matrix
        embedding_list = list(embeddings.values())
        entity_ids = list(embeddings.keys())
        
        if not embedding_list:
            return relationships
        
        # Convert embeddings to matrix for cosine similarity
        embedding_matrix = np.array(embedding_list)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embedding_matrix)
        
        # Threshold for relationship creation
        similarity_threshold = 0.7
        
        # Find relationships based on similarity
        for i, source_id in enumerate(entity_ids):
            for j, target_id in enumerate(entity_ids):
                if i != j:  # Avoid self-relationships
                    similarity = similarity_matrix[i, j]
                    
                    if similarity > similarity_threshold:
                        # Find corresponding entities
                        source_entity = next((e for e in entities if e.id == source_id), None)
                        target_entity = next((e for e in entities if e.id == target_id), None)
                        
                        if source_entity and target_entity:
                            # Determine relationship type based on entity types
                            rel_type = self._infer_relationship_type(source_entity, target_entity)
                            
                            if rel_type:
                                relationship = Relationship(
                                    id=hashlib.md5(f"{source_id}:{target_id}:semantic".encode()).hexdigest(),
                                    type=rel_type,
                                    source_id=source_id,
                                    target_id=target_id,
                                    confidence=float(similarity),
                                    properties=[
                                        {
                                            'name': 'semantic_similarity', 
                                            'value': str(similarity),
                                            'confidence': float(similarity)
                                        }
                                    ]
                                )
                                relationships.append(relationship)
        
        return relationships

    def _infer_relationship_type(self, source: Entity, target: Entity) -> Optional[RelationType]:
        """Infer relationship type based on entity types."""
        type_mapping = {
            (EntityType.CONCEPT, EntityType.THEORY): RelationType.RELATES_TO,
            (EntityType.THEORY, EntityType.PHENOMENON): RelationType.PREDICTS,
            (EntityType.TECHNOLOGY, EntityType.CONCEPT): RelationType.USES,
            (EntityType.EXPERIMENT, EntityType.THEORY): RelationType.PROVES,
            (EntityType.EQUATION, EntityType.CONCEPT): RelationType.MEASURES
        }
        
        # Check both direct and reversed mappings
        key = (source.type, target.type)
        reversed_key = (target.type, source.type)
        
        return type_mapping.get(key) or type_mapping.get(reversed_key)

    def _ensure_concrete_tokens(self, tokens):
        """
        Convert generator or iterable tokens to a concrete list of tokens.
        
        Args:
            tokens: A generator, iterator, or list of tokens
        
        Returns:
            A list of concrete token-like objects
        """
        try:
            # If it's already a list, return it
            if isinstance(tokens, list):
                return tokens
            
            # Try converting to list
            tokens_list = list(tokens)
            
            return tokens_list
        except Exception as e:
            logger.warning(f"Could not convert tokens to list: {str(e)}")
            return []

    async def _find_dependency_relationships(
        self, 
        entities: List[Entity], 
        texts: List[str]
    ) -> List[Relationship]:
        """
        Find dependency relationships between entities using advanced techniques.
        
        Args:
            entities: List of extracted entities
            texts: List of full text contexts for relationship analysis
        
        Returns:
            List of discovered relationships
        """
        relationships = []
        
        for text in texts:
            # Initialize confidence scorer
            confidence_scorer = RelationshipConfidenceScorer()
            
            # Prepare list to store relationships
            text_relationships = []
            
            # Perform pairwise relationship analysis
            for i, source_entity in enumerate(entities):
                for j, target_entity in enumerate(entities):
                    if i == j:  # Skip self-relationships
                        continue
                    
                    # Analyze text window around entities
                    entity_context = self._extract_entity_context(
                        source_entity, 
                        target_entity, 
                        text
                    )
                    
                    # Create potential relationship
                    relationship = Relationship(
                        source_id=source_entity.id,
                        target_id=target_entity.id,
                        type=RelationType.RELATES_TO,  # Default type
                        confidence=0.0
                    )
                    
                    # Calculate relationship confidence
                    confidence = confidence_scorer.calculate_relationship_confidence(
                        relationship, 
                        source_entity, 
                        target_entity, 
                        entity_context
                    )
                    
                    # Update relationship confidence
                    relationship.confidence = confidence
                    
                    # Add relationship if above confidence threshold
                    if confidence > 0.3:
                        text_relationships.append(relationship)
            
            # Refine relationships using advanced scoring
            refined_relationships = confidence_scorer.refine_graph_relationships(
                text_relationships, 
                entities, 
                text
            )
            
            relationships.extend(refined_relationships)
        
        return relationships

    def _extract_entity_context(
        self, 
        source_entity: Entity, 
        target_entity: Entity, 
        full_context: str, 
        window_size: int = 100
    ) -> str:
        """
        Extract contextual text around entities.
        
        Args:
            source_entity: Source entity
            target_entity: Target entity
            full_context: Full text context
            window_size: Number of characters around entities
        
        Returns:
            Extracted context text
        """
        # Find entity positions in context
        source_pos = full_context.lower().find(source_entity.name.lower())
        target_pos = full_context.lower().find(target_entity.name.lower())
        
        if source_pos == -1 or target_pos == -1:
            return full_context
        
        # Determine start and end of context window
        start = max(0, min(source_pos, target_pos) - window_size)
        end = min(len(full_context), max(source_pos, target_pos) + window_size)
        
        return full_context[start:end]

    def _find_citation_relationships(self, entities: List[Entity]) -> List[Relationship]:
        """Find citation-based relationships between entities."""
        relationships = []
        
        # Create citation graph based on references
        for source in entities:
            for target in entities:
                if source != target and set(source.references) & set(target.references):
                    relationship = Relationship(
                        id=hashlib.md5(f"{source.id}:{target.id}:citation".encode()).hexdigest(),
                        type=RelationType.RELATES_TO,
                        source_id=source.id,
                        target_id=target.id,
                        confidence=0.7,
                        properties=[
                            {
                                'name': 'shared_references', 
                                'value': str(set(source.references) & set(target.references)),
                                'confidence': 0.7
                            }
                        ]
                    )
                    relationships.append(relationship)
        
        return relationships

    def _validate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """Validate and deduplicate relationships."""
        # Remove duplicate relationships
        unique_relationships = []
        seen_pairs = set()
        
        for rel in relationships:
            pair_key = (rel.source_id, rel.target_id, rel.type)
            if pair_key not in seen_pairs and pair_key[::-1] not in seen_pairs:
                unique_relationships.append(rel)
                seen_pairs.add(pair_key)
        
        # Filter out low-confidence relationships
        valid_relationships = [
            rel for rel in unique_relationships 
            if rel.confidence > 0.5
        ]
        
        return valid_relationships
