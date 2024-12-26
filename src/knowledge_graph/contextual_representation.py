import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import scibert

# Local imports
from .schema import Entity, EntityType
from .config import CONFIG
from .logging_config import get_logger, log_performance

@dataclass
class ContextualRepresentationConfig:
    """
    Configuration for contextual representation learning.
    """
    domain: str = 'physics'
    embedding_dim: int = 768
    context_window: int = 5
    learning_rate: float = 1e-4
    dropout_rate: float = 0.1

class ContextualRepresentationLearner:
    """
    Advanced contextual representation learning system.
    
    Combines:
    - Domain-specific embedding techniques
    - Contextual feature extraction
    - Adaptive representation modeling
    """
    
    def __init__(self, config: ContextualRepresentationConfig = None):
        """
        Initialize contextual representation learner.
        
        Args:
            config: Contextual representation configuration
        """
        self.logger = get_logger(__name__)
        
        # Configuration
        self.config = config or ContextualRepresentationConfig()
        
        # Base embedding model
        self.base_model = self._load_base_embedding_model()
        
        # Contextual feature extractor
        self.context_encoder = self._build_context_encoder()
    
    def _load_base_embedding_model(self):
        """
        Load domain-specific base embedding model.
        
        Returns:
            Configured embedding model
        """
        try:
            # Use SciBERT for scientific domain
            model_name = 'allenai/scibert_scivocab_uncased'
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            return {
                'tokenizer': tokenizer,
                'model': model
            }
        
        except Exception as e:
            self.logger.error(f"Failed to load base embedding model: {e}")
            return None
    
    def _build_context_encoder(self):
        """
        Build contextual feature encoder.
        
        Returns:
            Contextual feature encoder neural network
        """
        class ContextEncoder(nn.Module):
            def __init__(self, input_dim, hidden_dim, dropout_rate):
                super().__init__()
                
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim, input_dim)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        return ContextEncoder(
            input_dim=self.config.embedding_dim,
            hidden_dim=self.config.embedding_dim * 2,
            dropout_rate=self.config.dropout_rate
        )
    
    @log_performance()
    async def generate_contextual_embeddings(
        self, 
        entities: List[Entity], 
        context: Optional[List[str]] = None
    ) -> List[np.ndarray]:
        """
        Generate contextual embeddings for entities.
        
        Args:
            entities: List of entities to embed
            context: Optional contextual information
        
        Returns:
            List of contextual embeddings
        """
        if not self.base_model:
            return [np.zeros(self.config.embedding_dim) for _ in entities]
        
        contextual_embeddings = []
        
        for entity in entities:
            # Tokenize entity name
            tokens = self.base_model['tokenizer'](
                entity.name, 
                return_tensors='pt', 
                truncation=True, 
                max_length=512
            )
            
            # Generate base embedding
            with torch.no_grad():
                base_embedding = self.base_model['model'](**tokens).last_hidden_state.mean(dim=1)
            
            # Apply contextual encoding if context provided
            if context:
                # Generate context embeddings
                context_tokens = self.base_model['tokenizer'](
                    context, 
                    return_tensors='pt', 
                    truncation=True, 
                    max_length=512,
                    padding=True
                )
                
                with torch.no_grad():
                    context_embeddings = self.base_model['model'](**context_tokens).last_hidden_state.mean(dim=1)
                
                # Contextual feature enhancement
                base_embedding = self.context_encoder(
                    torch.cat([base_embedding, context_embeddings.mean(dim=0)], dim=-1)
                )
            
            # Convert to numpy
            contextual_embedding = base_embedding.numpy().flatten()
            contextual_embeddings.append(contextual_embedding)
        
        return contextual_embeddings
    
    async def learn_domain_representation(
        self, 
        entities: List[Entity], 
        training_context: List[str]
    ) -> Dict[str, Any]:
        """
        Learn domain-specific representation from entities and context.
        
        Args:
            entities: List of domain entities
            training_context: Contextual training information
        
        Returns:
            Domain representation learning metrics
        """
        # Generate initial contextual embeddings
        embeddings = await self.generate_contextual_embeddings(
            entities, training_context
        )
        
        # Analyze embedding characteristics
        representation_metrics = {
            'embedding_dimensions': len(embeddings[0]),
            'unique_entities': len(set(entity.name for entity in entities)),
            'embedding_variance': np.var(embeddings, axis=0).mean(),
            'semantic_diversity': self._calculate_semantic_diversity(embeddings)
        }
        
        return representation_metrics
    
    def _calculate_semantic_diversity(self, embeddings: List[np.ndarray]) -> float:
        """
        Calculate semantic diversity of embeddings.
        
        Args:
            embeddings: List of embedding vectors
        
        Returns:
            Semantic diversity score
        """
        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append(similarity)
        
        # Diversity is inversely related to average similarity
        return 1 - np.mean(similarities) if similarities else 0.0

class DomainSpecificTokenizer:
    """
    Advanced tokenization system for scientific and technical domains.
    
    Provides:
    - Domain-specific vocabulary
    - Contextual token understanding
    - Sub-word tokenization
    """
    
    def __init__(self, domain: str = 'physics'):
        """
        Initialize domain-specific tokenizer.
        
        Args:
            domain: Specific scientific or technological domain
        """
        self.logger = get_logger(__name__)
        
        # Load domain-specific tokenizer
        self.tokenizer = self._load_domain_tokenizer()
        
        # Domain configuration
        self.domain = domain
    
    def _load_domain_tokenizer(self):
        """
        Load domain-specific tokenizer.
        
        Returns:
            Configured tokenizer
        """
        try:
            # Use SciBERT for scientific domain tokenization
            tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            
            # Add domain-specific tokens
            domain_tokens = [
                # Physics-specific tokens
                '[QUANTUM]', '[RELATIVITY]', '[PARTICLE]',
                '[WAVE_FUNCTION]', '[ENTANGLEMENT]',
                
                # Technology-specific tokens
                '[WARP_DRIVE]', '[PROPULSION]', '[ENERGY_SYSTEM]'
            ]
            
            tokenizer.add_tokens(domain_tokens)
            
            return tokenizer
        
        except Exception as e:
            self.logger.error(f"Failed to load domain tokenizer: {e}")
            return None
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize input text with domain-specific understanding.
        
        Args:
            text: Input text to tokenize
        
        Returns:
            List of tokens
        """
        if not self.tokenizer:
            return text.split()
        
        # Tokenize with domain-specific model
        tokens = self.tokenizer.tokenize(text)
        
        return tokens

# Factory function for creating contextual representation learners
def create_contextual_representation_learner(
    domain: str = 'physics', 
    config: Optional[ContextualRepresentationConfig] = None
) -> ContextualRepresentationLearner:
    """
    Create a domain-specific contextual representation learner.
    
    Args:
        domain: Scientific or technological domain
        config: Optional configuration override
    
    Returns:
        Configured ContextualRepresentationLearner
    """
    if config is None:
        config = ContextualRepresentationConfig(domain=domain)
    
    return ContextualRepresentationLearner(config)
