import pytest
import numpy as np
import networkx as nx
from typing import Dict, List, Set

from src.knowledge_fusion.advanced_knowledge_fusion import (
    AdvancedKnowledgeFusion,
    KnowledgeFragment
)
from src.knowledge_graph.distributed_knowledge_graph import DistributedKnowledgeGraphEngine
from src.semantic_understanding.semantic_intent_engine import SemanticIntentEngine
from src.reasoning.explainable_reasoning_engine import ExplainableReasoningEngine

class TestAdvancedKnowledgeFusion:
    @pytest.fixture
    def knowledge_fusion_engine(self):
        """
        Create a knowledge fusion engine with mock dependencies
        """
        # Create mock dependencies
        knowledge_graph = DistributedKnowledgeGraphEngine()
        semantic_engine = SemanticIntentEngine()
        reasoning_engine = ExplainableReasoningEngine()
        
        return AdvancedKnowledgeFusion(
            knowledge_graph,
            semantic_engine,
            reasoning_engine
        )
    
    @pytest.fixture
    def mock_knowledge_graph(self, knowledge_fusion_engine):
        """
        Create a mock knowledge graph for testing
        """
        # Create a sample knowledge graph
        G = nx.DiGraph()
        
        # Add sample nodes with domain information
        nodes = [
            {
                'id': 'quantum_computing',
                'domain': 'technology',
                'content': {'description': 'Advanced computing paradigm'},
                'embedding': np.random.rand(128)
            },
            {
                'id': 'machine_learning',
                'domain': 'artificial_intelligence',
                'content': {'description': 'Data-driven learning'},
                'embedding': np.random.rand(128)
            }
        ]
        
        for node in nodes:
            G.add_node(node['id'], **node)
        
        # Add sample edges
        G.add_edge('quantum_computing', 'machine_learning')
        
        # Mock method to return this graph
        knowledge_fusion_engine.knowledge_graph.extract_domain_subgraph = lambda domain: G
        
        return G
    
    @pytest.fixture
    def sample_knowledge_fragments(self) -> List[KnowledgeFragment]:
        """
        Create sample knowledge fragments for testing
        """
        fragments = []
        
        # Create fragments from different domains
        domains = ['technology', 'artificial_intelligence']
        
        for domain in domains:
            fragment = KnowledgeFragment(
                id=f"fragment_{domain}",
                domain=domain,
                content={'description': f'Sample {domain} content'},
                semantic_embedding=np.random.rand(128),
                relationships={
                    'incoming': [f'in_{domain}_1', f'in_{domain}_2'],
                    'outgoing': [f'out_{domain}_1', f'out_{domain}_2']
                },
                confidence=0.85,
                source_domains={domain}
            )
            fragments.append(fragment)
        
        return fragments
    
    def test_cross_domain_synthesis(
        self,
        knowledge_fusion_engine,
        mock_knowledge_graph,
        sample_knowledge_fragments
    ):
        """
        Test cross-domain knowledge synthesis
        
        Validates:
        - Knowledge extraction
        - Cross-domain relationship identification
        - Knowledge synthesis process
        """
        # Test domains
        source_domains = ['technology', 'artificial_intelligence']
        target_domain = 'interdisciplinary'
        
        # Synthesize cross-domain knowledge
        synthesized_fragments = knowledge_fusion_engine.synthesize_cross_domain_knowledge(
            source_domains,
            target_domain
        )
        
        # Validate synthesized fragments
        assert isinstance(synthesized_fragments, list), "Invalid synthesized fragments type"
        
        for fragment in synthesized_fragments:
            assert isinstance(fragment, KnowledgeFragment), "Invalid fragment type"
            
            # Check fragment properties
            assert fragment.id is not None, "Missing fragment ID"
            assert fragment.domain == target_domain, "Incorrect target domain"
            assert isinstance(fragment.content, dict), "Invalid content type"
            assert isinstance(fragment.semantic_embedding, np.ndarray), "Invalid embedding type"
            assert isinstance(fragment.relationships, dict), "Invalid relationships type"
            assert 0 <= fragment.confidence <= 1, "Invalid confidence score"
            assert len(fragment.source_domains) > 0, "Missing source domains"
    
    def test_knowledge_validation(
        self,
        knowledge_fusion_engine,
        sample_knowledge_fragments
    ):
        """
        Test knowledge validation process
        
        Validates:
        - Validation metrics computation
        - Validation criteria checking
        """
        # Validate knowledge fragments
        validated_fragments = knowledge_fusion_engine._validate_knowledge_fragments(
            sample_knowledge_fragments
        )
        
        # Check validation results
        assert isinstance(validated_fragments, list), "Invalid validated fragments type"
        
        for fragment in validated_fragments:
            # Check validation metrics
            assert isinstance(fragment.validation_metrics, dict), "Invalid validation metrics type"
            
            for metric_name in knowledge_fusion_engine.validation_metrics:
                assert metric_name in fragment.validation_metrics, f"Missing validation metric: {metric_name}"
                score = fragment.validation_metrics[metric_name]
                assert 0 <= score <= 1, f"Invalid validation score for {metric_name}"
    
    def test_fragment_similarity(
        self,
        knowledge_fusion_engine,
        sample_knowledge_fragments
    ):
        """
        Test fragment similarity computation
        
        Validates:
        - Similarity calculation
        - Threshold-based filtering
        """
        if len(sample_knowledge_fragments) < 2:
            return
        
        # Compute similarity between fragments
        fragment1, fragment2 = sample_knowledge_fragments[:2]
        similarity = knowledge_fusion_engine._compute_fragment_similarity(
            fragment1,
            fragment2
        )
        
        # Validate similarity score
        assert isinstance(similarity, float), "Invalid similarity score type"
        assert -1 <= similarity <= 1, "Similarity score out of range"
    
    def test_fragment_merging(
        self,
        knowledge_fusion_engine,
        sample_knowledge_fragments
    ):
        """
        Test fragment merging process
        
        Validates:
        - Content merging
        - Embedding merging
        - Relationship merging
        """
        if len(sample_knowledge_fragments) < 2:
            return
        
        # Create fragment group
        fragment_group = set(sample_knowledge_fragments[:2])
        
        # Test content merging
        merged_content = knowledge_fusion_engine._merge_fragment_content(fragment_group)
        assert isinstance(merged_content, dict), "Invalid merged content type"
        
        # Test embedding merging
        merged_embedding = knowledge_fusion_engine._compute_merged_embedding(fragment_group)
        assert isinstance(merged_embedding, np.ndarray), "Invalid merged embedding type"
        assert np.isclose(np.linalg.norm(merged_embedding), 1.0), "Merged embedding not normalized"
        
        # Test relationship merging
        merged_relationships = knowledge_fusion_engine._merge_relationships(fragment_group)
        assert isinstance(merged_relationships, dict), "Invalid merged relationships type"
        assert 'incoming' in merged_relationships, "Missing incoming relationships"
        assert 'outgoing' in merged_relationships, "Missing outgoing relationships"
    
    def test_validation_criteria(
        self,
        knowledge_fusion_engine
    ):
        """
        Test validation criteria checking
        
        Validates:
        - Validation score thresholds
        - Criteria combination
        """
        # Test cases with different validation scores
        test_cases = [
            {
                'semantic_consistency': 0.9,
                'structural_integrity': 0.85,
                'cross_domain_coherence': 0.88,
                'temporal_stability': 0.92
            },
            {
                'semantic_consistency': 0.7,
                'structural_integrity': 0.65,
                'cross_domain_coherence': 0.68,
                'temporal_stability': 0.72
            }
        ]
        
        for scores in test_cases:
            # Check validation criteria
            meets_criteria = knowledge_fusion_engine._meets_validation_criteria(scores)
            
            # Validate result
            assert isinstance(meets_criteria, bool), "Invalid validation result type"
            
            # Check if result matches expected threshold
            expected_result = all(
                score >= knowledge_fusion_engine.fusion_config['min_validation_score']
                for score in scores.values()
            )
            assert meets_criteria == expected_result, "Incorrect validation criteria check"
