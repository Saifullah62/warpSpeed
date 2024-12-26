import asyncio
from typing import List, Dict, Any, Optional, Union
from enum import Enum, auto
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Local imports
from .schema import Entity, EntityType, Relationship, RelationshipType
from .config import CONFIG
from .logging_config import get_logger, log_performance

class ReasoningStrategy(Enum):
    """
    Strategies for cross-domain reasoning.
    """
    ANALOGICAL = auto()
    PROBABILISTIC = auto()
    LOGICAL_INFERENCE = auto()
    HYPOTHESIS_GENERATION = auto()

class LogicalInferenceEngine:
    """
    Advanced logical inference engine for cross-domain reasoning.
    
    Provides mechanisms to:
    - Generate logical inferences
    - Map analogies across domains
    - Create probabilistic reasoning models
    """
    
    def __init__(self, domains: List[str] = ['physics', 'technology']):
        """
        Initialize logical inference engine.
        
        Args:
            domains: Scientific or technological domains to reason across
        """
        self.logger = get_logger(__name__)
        
        # Multi-domain embedding model
        self.embedding_model = self._load_multi_domain_embeddings(domains)
        
        # Reasoning configuration
        self.config = CONFIG.get_config_section('cross_domain_reasoning')
    
    def _load_multi_domain_embeddings(self, domains: List[str]):
        """
        Load multi-domain embedding models.
        
        Args:
            domains: List of domains to load embeddings for
        
        Returns:
            Multi-domain embedding models
        """
        try:
            # Use SciBERT as base model
            tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
            
            return {
                'tokenizer': tokenizer,
                'model': model
            }
        
        except Exception as e:
            self.logger.error(f"Failed to load multi-domain embeddings: {e}")
            return None
    
    @log_performance()
    async def generate_logical_inferences(
        self, 
        entities: List[Entity], 
        relationships: List[Relationship]
    ) -> List[Dict[str, Any]]:
        """
        Generate logical inferences across entities and relationships.
        
        Args:
            entities: List of entities to reason about
            relationships: Existing relationships
        
        Returns:
            List of logical inferences
        """
        inferences = []
        
        # Analyze entity relationships
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], start=i+1):
                # Check if entities are from different domains
                if entity1.type != entity2.type:
                    # Generate cross-domain inference
                    inference = self._generate_cross_domain_inference(
                        entity1, entity2, relationships
                    )
                    
                    if inference:
                        inferences.append(inference)
        
        return inferences
    
    def _generate_cross_domain_inference(
        self, 
        entity1: Entity, 
        entity2: Entity, 
        relationships: List[Relationship]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate inference between two entities from different domains.
        
        Args:
            entity1: First entity
            entity2: Second entity
            relationships: Existing relationships
        
        Returns:
            Cross-domain inference
        """
        # Calculate semantic similarity
        similarity = self._calculate_semantic_similarity(entity1, entity2)
        
        # Find potential relationships
        potential_relationships = [
            rel for rel in relationships 
            if (
                (rel.source == entity1 and rel.target == entity2) or
                (rel.source == entity2 and rel.target == entity1)
            )
        ]
        
        # Generate inference if similarity is high
        if similarity > 0.7:
            return {
                'entity1': entity1.name,
                'entity2': entity2.name,
                'similarity': similarity,
                'potential_relationships': [
                    rel.type.value for rel in potential_relationships
                ],
                'inference_type': 'cross_domain_analogy'
            }
        
        return None
    
    def _calculate_semantic_similarity(
        self, 
        entity1: Entity, 
        entity2: Entity
    ) -> float:
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
            # Tokenize entities
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
    
    async def generate_hypotheses(
        self, 
        entities: List[Entity]
    ) -> List[Dict[str, Any]]:
        """
        Generate hypotheses based on entity characteristics.
        
        Args:
            entities: List of entities to generate hypotheses about
        
        Returns:
            List of generated hypotheses
        """
        hypotheses = []
        
        # Group entities by type
        entity_groups = {}
        for entity in entities:
            if entity.type not in entity_groups:
                entity_groups[entity.type] = []
            entity_groups[entity.type].append(entity)
        
        # Generate hypotheses across entity types
        for type1 in entity_groups:
            for type2 in entity_groups:
                if type1 != type2:
                    hypothesis = self._generate_inter_type_hypothesis(
                        entity_groups[type1], 
                        entity_groups[type2]
                    )
                    
                    if hypothesis:
                        hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_inter_type_hypothesis(
        self, 
        entities1: List[Entity], 
        entities2: List[Entity]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate hypothesis between two entity type groups.
        
        Args:
            entities1: First group of entities
            entities2: Second group of entities
        
        Returns:
            Inter-type hypothesis
        """
        # Calculate average semantic similarity
        similarities = []
        for e1 in entities1:
            for e2 in entities2:
                similarities.append(self._calculate_semantic_similarity(e1, e2))
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        # Generate hypothesis if similarity is significant
        if avg_similarity > 0.6:
            return {
                'type1': entities1[0].type.name,
                'type2': entities2[0].type.name,
                'avg_similarity': avg_similarity,
                'hypothesis': f"Potential relationship between {entities1[0].type.name} and {entities2[0].type.name}"
            }
        
        return None

class CrossDomainReasoningSystem:
    """
    Comprehensive cross-domain reasoning system.
    
    Combines multiple reasoning strategies to:
    - Generate logical inferences
    - Create domain-bridging hypotheses
    - Provide explainable reasoning
    """
    
    def __init__(self, domains: List[str] = ['physics', 'technology']):
        """
        Initialize cross-domain reasoning system.
        
        Args:
            domains: Scientific or technological domains to reason across
        """
        self.logger = get_logger(__name__)
        
        # Logical inference engine
        self.inference_engine = LogicalInferenceEngine(domains)
    
    @log_performance()
    async def reason_across_domains(
        self, 
        entities: List[Entity], 
        relationships: List[Relationship],
        strategies: Optional[List[ReasoningStrategy]] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-domain reasoning.
        
        Args:
            entities: List of entities to reason about
            relationships: Existing relationships
            strategies: Reasoning strategies to apply
        
        Returns:
            Reasoning results
        """
        # Default strategies
        if strategies is None:
            strategies = [
                ReasoningStrategy.LOGICAL_INFERENCE,
                ReasoningStrategy.HYPOTHESIS_GENERATION
            ]
        
        # Reasoning results
        reasoning_results = {
            'logical_inferences': [],
            'hypotheses': []
        }
        
        # Apply reasoning strategies
        if ReasoningStrategy.LOGICAL_INFERENCE in strategies:
            reasoning_results['logical_inferences'] = await self.inference_engine.generate_logical_inferences(
                entities, relationships
            )
        
        if ReasoningStrategy.HYPOTHESIS_GENERATION in strategies:
            reasoning_results['hypotheses'] = await self.inference_engine.generate_hypotheses(
                entities
            )
        
        return reasoning_results

# Factory function for creating cross-domain reasoning systems
def create_cross_domain_reasoning_system(
    domains: List[str] = ['physics', 'technology']
) -> CrossDomainReasoningSystem:
    """
    Create a cross-domain reasoning system.
    
    Args:
        domains: Scientific or technological domains to reason across
    
    Returns:
        Configured CrossDomainReasoningSystem
    """
    return CrossDomainReasoningSystem(domains)
