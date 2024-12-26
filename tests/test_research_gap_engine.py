import pytest
import numpy as np
import networkx as nx

from src.research_generation.research_gap_engine import (
    ResearchGapEngine,
    ResearchProposal
)
from src.knowledge_graph.distributed_knowledge_graph import DistributedKnowledgeGraphEngine
from src.semantic_understanding.semantic_intent_engine import SemanticIntentEngine

class TestResearchGapEngine:
    @pytest.fixture
    def research_gap_engine(self):
        """
        Create a research gap engine with mock dependencies
        """
        # Create mock knowledge graph
        knowledge_graph = DistributedKnowledgeGraphEngine()
        
        # Create mock semantic engine
        semantic_engine = SemanticIntentEngine()
        
        return ResearchGapEngine(
            knowledge_graph, 
            semantic_engine
        )
    
    @pytest.fixture
    def mock_knowledge_graph(self, research_gap_engine):
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
                'description': 'Advanced computing paradigm',
                'research_maturity': 'emerging'
            },
            {
                'id': 'machine_learning', 
                'domain': 'artificial_intelligence', 
                'description': 'Data-driven learning algorithms',
                'research_maturity': 'established'
            },
            {
                'id': 'neural_networks', 
                'domain': 'artificial_intelligence', 
                'description': 'Computational models inspired by brain structure',
                'research_maturity': 'mature'
            }
        ]
        
        for node in nodes:
            G.add_node(node['id'], **node)
        
        # Add sample edges
        edges = [
            ('quantum_computing', 'machine_learning'),
            ('machine_learning', 'neural_networks')
        ]
        
        G.add_edges_from(edges)
        
        # Mock method to return this graph
        research_gap_engine.knowledge_graph.extract_domain_subgraph = lambda domain: G
        
        return G
    
    def test_identify_research_gaps(
        self, 
        research_gap_engine, 
        mock_knowledge_graph
    ):
        """
        Test research gap identification process
        
        Validates:
        - Gap identification mechanism
        - Proposal generation
        - Domain-specific analysis
        """
        # Test different domains and max proposals
        test_cases = [
            {'domain': 'technology', 'max_proposals': 3},
            {'domain': 'artificial_intelligence', 'max_proposals': 2},
            {'domain': None, 'max_proposals': 5}
        ]
        
        for case in test_cases:
            # Identify research gaps
            research_proposals = research_gap_engine.identify_research_gaps(
                domain=case['domain'],
                max_proposals=case['max_proposals']
            )
            
            # Validate research proposals
            assert isinstance(research_proposals, list), "Invalid research proposals type"
            assert 0 < len(research_proposals) <= case['max_proposals'], "Incorrect number of proposals"
            
            # Validate individual proposals
            for proposal in research_proposals:
                assert isinstance(proposal, ResearchProposal), "Invalid proposal type"
                
                # Check proposal properties
                assert proposal.id is not None, "Missing proposal ID"
                assert proposal.title is not None, "Missing proposal title"
                assert proposal.domain is not None, "Missing proposal domain"
                
                # Validate priority and scores
                assert 0 <= proposal.priority <= 1, "Invalid proposal priority"
                assert 0 <= proposal.novelty_score <= 1, "Invalid novelty score"
                assert 0 <= proposal.interdisciplinary_potential <= 1, "Invalid interdisciplinary potential"
                
                # Check research questions and methodologies
                assert len(proposal.research_questions) > 0, "No research questions generated"
                assert len(proposal.proposed_methodologies) > 0, "No methodologies proposed"
                
                # Validate confidence interval
                lower, upper = proposal.confidence_interval
                assert 0 <= lower <= upper <= 1, "Invalid confidence interval"
    
    def test_research_gap_analysis(
        self, 
        research_gap_engine, 
        mock_knowledge_graph
    ):
        """
        Test detailed research gap analysis
        
        Validates:
        - Gap detection mechanism
        - Centrality-based gap identification
        """
        # Test domains
        test_domains = ['technology', 'artificial_intelligence']
        
        for domain in test_domains:
            # Analyze domain-specific research gaps
            research_gaps = research_gap_engine._analyze_domain_gaps(domain)
            
            # Validate research gaps
            assert isinstance(research_gaps, list), "Invalid research gaps type"
            
            # Check individual gap properties
            for gap in research_gaps:
                assert 'node' in gap, "Missing node in research gap"
                assert 'centrality' in gap, "Missing centrality in research gap"
                assert 'metadata' in gap, "Missing metadata in research gap"
                
                # Validate centrality
                assert 0 <= gap['centrality'] <= 1, "Invalid centrality score"
    
    def test_novelty_score_computation(
        self, 
        research_gap_engine, 
        mock_knowledge_graph
    ):
        """
        Test novelty score computation
        
        Validates:
        - Novelty score calculation
        - Factors influencing novelty
        """
        # Test research gaps
        test_gaps = [
            {
                'node': 'quantum_computing',
                'metadata': {'research_maturity': 'emerging'},
                'centrality': 0.2
            },
            {
                'node': 'machine_learning',
                'metadata': {'research_maturity': 'established'},
                'centrality': 0.5
            }
        ]
        
        for gap in test_gaps:
            # Compute novelty score
            novelty_score = research_gap_engine._compute_novelty_score(gap)
            
            # Validate novelty score
            assert isinstance(novelty_score, float), "Invalid novelty score type"
            assert 0 <= novelty_score <= 1, "Novelty score out of range"
    
    def test_interdisciplinary_potential(
        self, 
        research_gap_engine, 
        mock_knowledge_graph
    ):
        """
        Test interdisciplinary potential assessment
        
        Validates:
        - Interdisciplinary potential computation
        - Keyword-based assessment
        """
        # Test research gaps with different metadata
        test_gaps = [
            {
                'node': 'quantum_computing',
                'metadata': {
                    'description': 'Cross-domain quantum computing research',
                    'keywords': ['interdisciplinary', 'hybrid']
                }
            },
            {
                'node': 'machine_learning',
                'metadata': {
                    'description': 'Traditional machine learning approach',
                    'keywords': ['classical', 'standard']
                }
            }
        ]
        
        for gap in test_gaps:
            # Assess interdisciplinary potential
            interdisciplinary_score = research_gap_engine._assess_interdisciplinary_potential(gap)
            
            # Validate interdisciplinary score
            assert isinstance(interdisciplinary_score, float), "Invalid interdisciplinary score type"
            assert 0 <= interdisciplinary_score <= 1, "Interdisciplinary score out of range"
    
    def test_research_proposal_generation(
        self, 
        research_gap_engine, 
        mock_knowledge_graph
    ):
        """
        Test research proposal generation
        
        Validates:
        - Proposal generation mechanism
        - Research question and methodology generation
        """
        # Test domains and research gaps
        test_domains = ['technology', 'artificial_intelligence']
        
        for domain in test_domains:
            # Analyze domain-specific research gaps
            research_gaps = research_gap_engine._analyze_domain_gaps(domain)
            
            # Generate research proposals
            research_proposals = research_gap_engine._generate_research_proposals(
                domain, 
                research_gaps
            )
            
            # Validate research proposals
            assert isinstance(research_proposals, list), "Invalid research proposals type"
            
            # Check individual proposals
            for proposal in research_proposals:
                # Validate research questions
                assert len(proposal.research_questions) > 0, "No research questions generated"
                assert all(isinstance(q, str) for q in proposal.research_questions), "Invalid research questions"
                
                # Validate proposed methodologies
                assert len(proposal.proposed_methodologies) > 0, "No methodologies proposed"
                assert all(isinstance(m, str) for m in proposal.proposed_methodologies), "Invalid methodologies"
    
    def test_research_proposal_document(
        self, 
        research_gap_engine, 
        mock_knowledge_graph
    ):
        """
        Test comprehensive research proposal document generation
        
        Validates:
        - Proposal document structure
        - Detailed proposal generation
        """
        # Generate sample research proposals
        research_proposals = research_gap_engine.identify_research_gaps(max_proposals=2)
        
        for proposal in research_proposals:
            # Generate proposal document
            proposal_document = research_gap_engine.generate_research_proposal(proposal)
            
            # Validate proposal document structure
            assert isinstance(proposal_document, dict), "Invalid proposal document type"
            
            # Check key components
            assert 'id' in proposal_document, "Missing proposal ID"
            assert 'title' in proposal_document, "Missing proposal title"
            assert 'abstract' in proposal_document, "Missing proposal abstract"
            assert 'research_context' in proposal_document, "Missing research context"
            assert 'methodology' in proposal_document, "Missing methodology section"
            assert 'expected_outcomes' in proposal_document, "Missing expected outcomes"
            assert 'impact_assessment' in proposal_document, "Missing impact assessment"
            assert 'confidence_interval' in proposal_document, "Missing confidence interval"
            
            # Validate methodology section
            methodology = proposal_document['methodology']
            assert 'research_questions' in methodology, "Missing research questions"
            assert 'proposed_methodologies' in methodology, "Missing proposed methodologies"
    
    def test_research_priority_computation(
        self, 
        research_gap_engine
    ):
        """
        Test research priority computation
        
        Validates:
        - Priority calculation mechanism
        - Impact potential assessment
        """
        # Test various novelty and interdisciplinary scores
        test_cases = [
            (0.5, 0.5),
            (0.8, 0.2),
            (0.3, 0.7)
        ]
        
        for novelty_score, interdisciplinary_potential in test_cases:
            # Compute research priority
            priority, potential_impact = research_gap_engine._compute_research_priority(
                novelty_score, 
                interdisciplinary_potential
            )
            
            # Validate priority
            assert isinstance(priority, float), "Invalid priority type"
            assert 0 <= priority <= 1, "Priority out of range"
            
            # Validate potential impact
            assert isinstance(potential_impact, dict), "Invalid potential impact type"
            assert all(0 <= value <= 1 for value in potential_impact.values()), "Impact values out of range"
            
            # Check impact categories
            expected_categories = [
                'scientific_advancement',
                'technological_innovation',
                'economic_potential',
                'societal_impact'
            ]
            assert all(cat in potential_impact for cat in expected_categories), "Missing impact categories"
