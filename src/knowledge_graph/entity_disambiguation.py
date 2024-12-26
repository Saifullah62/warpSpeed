import asyncio
from typing import List, Dict, Any, Optional
from enum import Enum, auto

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import networkx as nx

# Local imports
from .schema import Entity, EntityType
from .config import CONFIG
from .logging_config import get_logger, log_performance

class DisambiguationStrategy(Enum):
    """
    Strategies for entity disambiguation.
    """
    CONTEXTUAL = auto()
    PROBABILISTIC = auto()
    SEMANTIC = auto()
    CROSS_REFERENCE = auto()

class DomainKnowledgeBase:
    """
    Domain-specific knowledge base for entity disambiguation.
    
    Provides contextual and semantic information to resolve 
    entity ambiguities in scientific and technological domains.
    """
    
    def __init__(self, domain: str = 'physics'):
        """
        Initialize domain-specific knowledge base.
        
        Args:
            domain: Specific scientific or technological domain
        """
        self.logger = get_logger(__name__)
        self.domain = domain
        
        # Load domain-specific embeddings
        self.embedding_model = self._load_domain_embeddings()
        
        # Initialize knowledge graph for relationship tracking
        self.knowledge_graph = nx.DiGraph()
    
    def _load_domain_embeddings(self):
        """
        Load domain-specific embedding model.
        
        Returns:
            Transformer-based embedding model
        """
        try:
            model_name = 'allenai/scibert_scivocab_uncased'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            return {
                'tokenizer': tokenizer,
                'model': model
            }
        except Exception as e:
            self.logger.error(f"Failed to load domain embeddings: {e}")
            return None
    
    def calculate_semantic_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """
        Calculate semantic similarity between two entities.
        
        Args:
            entity1: First entity
            entity2: Second entity
        
        Returns:
            Semantic similarity score
        """
        if not self.embedding_model:
            return 0.0
        
        try:
            # Tokenize and embed entities
            tokens1 = self.embedding_model['tokenizer'](
                entity1.name, return_tensors='pt'
            )
            tokens2 = self.embedding_model['tokenizer'](
                entity2.name, return_tensors='pt'
            )
            
            # Generate embeddings
            with torch.no_grad():
                embeddings1 = self.embedding_model['model'](**tokens1).last_hidden_state.mean(dim=1)
                embeddings2 = self.embedding_model['model'](**tokens2).last_hidden_state.mean(dim=1)
            
            # Calculate cosine similarity
            similarity = torch.nn.functional.cosine_similarity(
                embeddings1, embeddings2
            ).item()
            
            return max(0, min(1, (similarity + 1) / 2))
        
        except Exception as e:
            self.logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.0

class EntityDisambiguator:
    """
    Advanced entity disambiguation system.
    
    Combines multiple strategies to resolve entity ambiguities:
    - Contextual disambiguation
    - Probabilistic reasoning
    - Semantic similarity
    - Cross-reference validation
    """
    
    def __init__(self, domain: str = 'physics'):
        """
        Initialize entity disambiguator.
        
        Args:
            domain: Scientific or technological domain
        """
        self.logger = get_logger(__name__)
        self.domain_kb = DomainKnowledgeBase(domain)
        
        # Disambiguation configuration
        self.config = CONFIG.get_config_section('entity_extraction')
    
    @log_performance()
    async def disambiguate_entities(
        self, 
        entities: List[Entity], 
        strategies: Optional[List[DisambiguationStrategy]] = None
    ) -> List[Entity]:
        """
        Disambiguate a list of entities using multiple strategies.
        
        Args:
            entities: List of entities to disambiguate
            strategies: Disambiguation strategies to apply
        
        Returns:
            Disambiguated list of entities
        """
        # Default strategies
        if strategies is None:
            strategies = [
                DisambiguationStrategy.CONTEXTUAL,
                DisambiguationStrategy.SEMANTIC,
                DisambiguationStrategy.CROSS_REFERENCE
            ]
        
        # Parallel disambiguation tasks
        tasks = []
        
        # Apply each strategy
        if DisambiguationStrategy.SEMANTIC in strategies:
            tasks.append(self._semantic_disambiguation(entities))
        
        if DisambiguationStrategy.CONTEXTUAL in strategies:
            tasks.append(self._contextual_disambiguation(entities))
        
        if DisambiguationStrategy.CROSS_REFERENCE in strategies:
            tasks.append(self._cross_reference_disambiguation(entities))
        
        # Combine disambiguation results
        disambiguated_results = await asyncio.gather(*tasks)
        
        # Merge and reconcile results
        final_entities = self._reconcile_disambiguated_entities(
            disambiguated_results
        )
        
        return final_entities
    
    async def _semantic_disambiguation(self, entities: List[Entity]) -> List[Entity]:
        """
        Disambiguate entities using semantic similarity.
        
        Args:
            entities: List of entities to disambiguate
        
        Returns:
            Semantically disambiguated entities
        """
        disambiguated = []
        
        for i, entity in enumerate(entities):
            # Compare with other entities
            best_match = entity
            max_similarity = 0
            
            for j, other_entity in enumerate(entities):
                if i != j and entity.type == other_entity.type:
                    similarity = self.domain_kb.calculate_semantic_similarity(
                        entity, other_entity
                    )
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match = other_entity
            
            # Update entity with semantic context
            best_match.properties['semantic_similarity'] = max_similarity
            disambiguated.append(best_match)
        
        return disambiguated
    
    async def _contextual_disambiguation(self, entities: List[Entity]) -> List[Entity]:
        """
        Disambiguate entities using contextual information.
        
        Args:
            entities: List of entities to disambiguate
        
        Returns:
            Contextually disambiguated entities
        """
        # Placeholder for advanced contextual disambiguation
        # Would involve analyzing surrounding text, domain context, etc.
        return entities
    
    async def _cross_reference_disambiguation(self, entities: List[Entity]) -> List[Entity]:
        """
        Disambiguate entities by cross-referencing multiple sources.
        
        Args:
            entities: List of entities to disambiguate
        
        Returns:
            Cross-referenced disambiguated entities
        """
        # Placeholder for multi-source entity reconciliation
        return entities
    
    def _reconcile_disambiguated_entities(
        self, 
        disambiguated_results: List[List[Entity]]
    ) -> List[Entity]:
        """
        Reconcile disambiguated entities from multiple strategies.
        
        Args:
            disambiguated_results: Results from different disambiguation strategies
        
        Returns:
            Final reconciled list of entities
        """
        # Merge entities, prioritizing most confident results
        final_entities = {}
        
        for result_set in disambiguated_results:
            for entity in result_set:
                key = (entity.name, entity.type)
                
                # Keep entity with highest confidence/similarity
                if key not in final_entities or (
                    entity.properties.get('semantic_similarity', 0) > 
                    final_entities[key].properties.get('semantic_similarity', 0)
                ):
                    final_entities[key] = entity
        
        return list(final_entities.values())

# Optional: Factory for creating domain-specific disambiguators
def create_disambiguator(domain: str = 'physics') -> EntityDisambiguator:
    """
    Create a domain-specific entity disambiguator.
    
    Args:
        domain: Scientific or technological domain
    
    Returns:
        Configured EntityDisambiguator
    """
    return EntityDisambiguator(domain)
