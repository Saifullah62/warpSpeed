import pytest
import torch
import numpy as np
from typing import Dict, List, Any, Tuple

from src.semantic_understanding.multilingual_semantic_engine import (
    MultilingualSemanticEngine,
    SemanticRepresentation,
    CrossLingualAlignment
)

class TestMultilingualSemanticEngine:
    @pytest.fixture
    def semantic_engine(self):
        """Create semantic engine instance for testing"""
        return MultilingualSemanticEngine()
    
    @pytest.fixture
    def sample_texts(self) -> Dict[str, List[str]]:
        """Create sample texts in different languages"""
        return {
            'en': [
                "Quantum computing leverages quantum mechanics for computation",
                "Machine learning algorithms learn patterns from data"
            ],
            'es': [
                "La computación cuántica aprovecha la mecánica cuántica para la computación",
                "Los algoritmos de aprendizaje automático aprenden patrones de los datos"
            ],
            'fr': [
                "L'informatique quantique exploite la mécanique quantique pour le calcul",
                "Les algorithmes d'apprentissage automatique apprennent des motifs à partir des données"
            ]
        }
    
    def test_text_analysis(self, semantic_engine, sample_texts):
        """Test multilingual text analysis"""
        # Analyze English text
        representation = semantic_engine.analyze_text(
            text=sample_texts['en'][0],
            source_lang='en'
        )
        
        # Verify representation
        assert isinstance(representation, SemanticRepresentation)
        assert representation.text == sample_texts['en'][0]
        assert representation.language == 'en'
        assert isinstance(representation.embedding, torch.Tensor)
        assert len(representation.concepts) > 0
        assert len(representation.relations) > 0
        assert 0 <= representation.confidence <= 1
    
    def test_cross_lingual_analysis(self, semantic_engine, sample_texts):
        """Test cross-lingual analysis"""
        # Analyze with target languages
        representation = semantic_engine.analyze_text(
            text=sample_texts['en'][0],
            source_lang='en',
            target_langs=['es', 'fr']
        )
        
        # Verify cross-lingual analysis
        assert 'cross_lingual' in representation.metadata
        assert 'es' in representation.metadata['cross_lingual']
        assert 'fr' in representation.metadata['cross_lingual']
        
        for lang_analysis in representation.metadata['cross_lingual'].values():
            assert 'alignment_score' in lang_analysis
            assert 'semantic_drift' in lang_analysis
            assert 'confidence' in lang_analysis
    
    def test_language_alignment(self, semantic_engine, sample_texts):
        """Test language alignment"""
        alignment = semantic_engine.align_languages(
            source_texts=sample_texts['en'],
            target_texts=sample_texts['es'],
            source_lang='en',
            target_lang='es'
        )
        
        # Verify alignment
        assert isinstance(alignment, CrossLingualAlignment)
        assert alignment.source_lang == 'en'
        assert alignment.target_lang == 'es'
        assert 0 <= alignment.alignment_score <= 1
        assert len(alignment.translation_pairs) > 0
        assert 0 <= alignment.semantic_drift <= 1
        assert 0 <= alignment.confidence <= 1
    
    def test_knowledge_transfer(self, semantic_engine, sample_texts):
        """Test semantic knowledge transfer"""
        # Create source representation
        source_repr = semantic_engine.analyze_text(
            text=sample_texts['en'][0],
            source_lang='en'
        )
        
        # Transfer to target language
        transferred_repr = semantic_engine.transfer_knowledge(
            source_representation=source_repr,
            target_lang='es'
        )
        
        # Verify transfer
        assert isinstance(transferred_repr, SemanticRepresentation)
        assert transferred_repr.language == 'es'
        assert len(transferred_repr.concepts) > 0
        assert len(transferred_repr.relations) > 0
        assert transferred_repr.confidence <= source_repr.confidence
    
    def test_embedding_computation(self, semantic_engine, sample_texts):
        """Test embedding computation"""
        encoding = semantic_engine._encode_text(
            text=sample_texts['en'][0],
            language='en'
        )
        
        # Verify encoding
        assert 'embedding' in encoding
        assert 'attention_weights' in encoding
        assert 'confidence' in encoding
        assert isinstance(encoding['embedding'], torch.Tensor)
    
    def test_batch_processing(self, semantic_engine, sample_texts):
        """Test batch text processing"""
        embeddings = semantic_engine._batch_encode_texts(
            texts=sample_texts['en'],
            language='en'
        )
        
        # Verify batch embeddings
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[0] == len(sample_texts['en'])
    
    def test_concept_extraction(self, semantic_engine, sample_texts):
        """Test concept extraction"""
        # Encode text
        encoding = semantic_engine._encode_text(
            text=sample_texts['en'][0],
            language='en'
        )
        
        # Extract concepts
        concepts = semantic_engine._extract_concepts(
            encoding=encoding,
            language='en'
        )
        
        # Verify concepts
        assert isinstance(concepts, list)
        assert len(concepts) > 0
        assert all(isinstance(c, str) for c in concepts)
    
    def test_relation_extraction(self, semantic_engine, sample_texts):
        """Test relation extraction"""
        # Encode text
        encoding = semantic_engine._encode_text(
            text=sample_texts['en'][0],
            language='en'
        )
        
        # Extract concepts
        concepts = semantic_engine._extract_concepts(
            encoding=encoding,
            language='en'
        )
        
        # Extract relations
        relations = semantic_engine._extract_relations(
            encoding=encoding,
            concepts=concepts
        )
        
        # Verify relations
        assert isinstance(relations, list)
        assert all(isinstance(r, tuple) and len(r) == 3 for r in relations)
    
    def test_semantic_drift(self, semantic_engine):
        """Test semantic drift computation"""
        # Create sample embeddings
        source_embeddings = torch.randn(5, 768)
        target_embeddings = torch.randn(5, 768)
        
        # Compute drift
        drift = semantic_engine._compute_semantic_drift(
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings
        )
        
        # Verify drift
        assert isinstance(drift, float)
        assert drift >= 0
    
    def test_translation_pair_finding(self, semantic_engine, sample_texts):
        """Test translation pair finding"""
        # Create embeddings
        source_embeddings = semantic_engine._batch_encode_texts(
            texts=sample_texts['en'],
            language='en'
        )
        target_embeddings = semantic_engine._batch_encode_texts(
            texts=sample_texts['es'],
            language='es'
        )
        
        # Find pairs
        pairs = semantic_engine._find_translation_pairs(
            source_texts=sample_texts['en'],
            target_texts=sample_texts['es'],
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings
        )
        
        # Verify pairs
        assert isinstance(pairs, list)
        assert len(pairs) > 0
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
    
    def test_error_handling(self, semantic_engine):
        """Test error handling for unsupported languages"""
        with pytest.raises(ValueError):
            semantic_engine.analyze_text(
                text="Sample text",
                source_lang="unsupported_lang"
            )
        
        with pytest.raises(ValueError):
            semantic_engine.transfer_knowledge(
                source_representation=SemanticRepresentation(
                    text="",
                    language="en",
                    embedding=torch.randn(768),
                    concepts=[],
                    relations=[],
                    confidence=1.0
                ),
                target_lang="unsupported_lang"
            )
