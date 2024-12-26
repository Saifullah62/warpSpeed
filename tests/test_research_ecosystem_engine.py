import pytest
import numpy as np
import networkx as nx

from src.research_generation.research_ecosystem_engine import (
    ResearchEcosystemEngine,
    ResearchDomain
)
from src.knowledge_graph.distributed_knowledge_graph import DistributedKnowledgeGraphEngine
from src.research_generation.research_gap_engine import ResearchGapEngine
from src.semantic_understanding.semantic_intent_engine import SemanticIntentEngine

class TestResearchEcosystemEngine:
    @pytest.fixture
    def research_ecosystem_engine(self):
        """
        Create a research ecosystem engine with mock dependencies
        """
        # Create mock dependencies
        knowledge_graph = DistributedKnowledgeGraphEngine()
        research_gap_engine = ResearchGapEngine()
        semantic_engine = SemanticIntentEngine()
        
        return ResearchEcosystemEngine(
            knowledge_graph, 
            research_gap_engine, 
            semantic_engine
        )
    
    @pytest.fixture
    def mock_knowledge_graph(self, research_ecosystem_engine):
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
                'attributes': {'cross-domain': True}
            },
            {
                'id': 'machine_learning', 
                'domain': 'artificial_intelligence', 
                'attributes': {'cross-domain': False}
            },
            {
                'id': 'neural_networks', 
                'domain': 'artificial_intelligence', 
                'attributes': {'cross-domain': True}
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
        research_ecosystem_engine.knowledge_graph.extract_domain_subgraph = lambda domain: G
        
        return G
    
    def test_model_research_ecosystem(
        self, 
        research_ecosystem_engine, 
        mock_knowledge_graph
    ):
        """
        Test research ecosystem modeling
        
        Validates:
        - Ecosystem modeling process
        - Domain evolution
        - Time horizon handling
        """
        # Test different time horizons and granularities
        test_cases = [
            {'time_horizon': 3, 'granularity': 'yearly'},
            {'time_horizon': 4, 'granularity': 'quarterly'},
            {'time_horizon': 5, 'granularity': 'yearly'}
        ]
        
        for case in test_cases:
            # Model research ecosystem
            research_domains = research_ecosystem_engine.model_research_ecosystem(
                time_horizon=case['time_horizon'],
                granularity=case['granularity']
            )
            
            # Validate research domains
            assert isinstance(research_domains, list), "Invalid research domains type"
            assert len(research_domains) > 0, "No research domains generated"
            
            # Check individual domains
            for domain in research_domains:
                assert isinstance(domain, ResearchDomain), "Invalid domain type"
                
                # Check domain properties
                assert domain.name is not None, "Missing domain name"
                assert 0 <= domain.complexity <= 10, "Invalid domain complexity"
                assert 0 <= domain.innovation_rate <= 1, "Invalid innovation rate"
                assert 0 <= domain.interdisciplinary_potential <= 1, "Invalid interdisciplinary potential"
                assert 0 <= domain.knowledge_graph_density <= 1, "Invalid knowledge graph density"
                
                # Check temporal dynamics
                assert isinstance(domain.temporal_dynamics, dict), "Invalid temporal dynamics type"
                assert len(domain.temporal_dynamics) > 0, "No temporal dynamics recorded"
    
    def test_domain_evolution(
        self, 
        research_ecosystem_engine, 
        mock_knowledge_graph
    ):
        """
        Test domain evolution process
        
        Validates:
        - Domain characteristic updates
        - Evolution parameters
        - Temporal dynamics
        """
        # Initialize research domains
        initial_domains = research_ecosystem_engine._initialize_research_domains()
        
        # Evolve domains
        evolved_domains = research_ecosystem_engine._evolve_research_domains(
            initial_domains, 
            period=1, 
            granularity='yearly'
        )
        
        # Validate domain evolution
        assert len(evolved_domains) == len(initial_domains), "Domain count mismatch"
        
        for initial, evolved in zip(initial_domains, evolved_domains):
            # Check domain property evolution
            assert evolved.complexity >= initial.complexity, "Invalid complexity evolution"
            assert evolved.innovation_rate <= initial.innovation_rate, "Invalid innovation rate evolution"
            assert evolved.interdisciplinary_potential >= initial.interdisciplinary_potential, "Invalid interdisciplinary potential evolution"
            
            # Check temporal dynamics updates
            assert len(evolved.temporal_dynamics) > len(initial.temporal_dynamics), "Temporal dynamics not updated"
    
    def test_research_trajectory_report(
        self, 
        research_ecosystem_engine, 
        mock_knowledge_graph
    ):
        """
        Test research trajectory report generation
        
        Validates:
        - Report structure
        - Metrics computation
        - Recommendations generation
        """
        # Generate research domains
        research_domains = research_ecosystem_engine.model_research_ecosystem(
            time_horizon=3,
            granularity='yearly'
        )
        
        # Generate trajectory report
        report = research_ecosystem_engine.generate_research_trajectory_report(research_domains)
        
        # Validate report structure
        assert isinstance(report, dict), "Invalid report type"
        assert 'overall_ecosystem_metrics' in report, "Missing ecosystem metrics"
        assert 'domain_evolution_details' in report, "Missing domain evolution details"
        assert 'interdisciplinary_trends' in report, "Missing interdisciplinary trends"
        assert 'research_strategy_recommendations' in report, "Missing research recommendations"
        
        # Check ecosystem metrics
        metrics = report['overall_ecosystem_metrics']
        assert 0 <= metrics['average_complexity'] <= 10, "Invalid average complexity"
        assert 0 <= metrics['average_innovation_rate'] <= 1, "Invalid average innovation rate"
        assert 0 <= metrics['average_interdisciplinary_potential'] <= 1, "Invalid average interdisciplinary potential"
        
        # Check domain evolution details
        domain_details = report['domain_evolution_details']
        assert len(domain_details) > 0, "No domain evolution details"
        
        for domain_name, details in domain_details.items():
            assert isinstance(details, dict), "Invalid domain details type"
            assert 'complexity_trajectory' in details, "Missing complexity trajectory"
            assert 'innovation_trajectory' in details, "Missing innovation trajectory"
            assert 'interdisciplinary_trajectory' in details, "Missing interdisciplinary trajectory"
    
    def test_interdisciplinary_trends(
        self, 
        research_ecosystem_engine, 
        mock_knowledge_graph
    ):
        """
        Test interdisciplinary trend analysis
        
        Validates:
        - Trend computation
        - Cross-domain potential assessment
        - High potential domain identification
        """
        # Generate research domains
        research_domains = research_ecosystem_engine.model_research_ecosystem(
            time_horizon=3,
            granularity='yearly'
        )
        
        # Analyze interdisciplinary trends
        trends = research_ecosystem_engine._analyze_interdisciplinary_trends(research_domains)
        
        # Validate trend analysis
        assert isinstance(trends, dict), "Invalid trends type"
        assert 'cross_domain_potential' in trends, "Missing cross-domain potential"
        assert 'high_potential_domains' in trends, "Missing high potential domains"
        
        # Check trend metrics
        assert 0 <= trends['cross_domain_potential'] <= 1, "Invalid cross-domain potential"
        assert isinstance(trends['high_potential_domains'], list), "Invalid high potential domains type"
    
    def test_research_recommendations(
        self, 
        research_ecosystem_engine, 
        mock_knowledge_graph
    ):
        """
        Test research recommendation generation
        
        Validates:
        - Recommendation generation
        - Recommendation structure
        - Recommendation rationale
        """
        # Generate research domains
        research_domains = research_ecosystem_engine.model_research_ecosystem(
            time_horizon=3,
            granularity='yearly'
        )
        
        # Generate recommendations
        recommendations = research_ecosystem_engine._generate_research_recommendations(research_domains)
        
        # Validate recommendations
        assert isinstance(recommendations, list), "Invalid recommendations type"
        
        for recommendation in recommendations:
            assert isinstance(recommendation, dict), "Invalid recommendation type"
            assert 'domain' in recommendation, "Missing domain in recommendation"
            assert 'recommendation' in recommendation, "Missing recommendation text"
            assert 'rationale' in recommendation, "Missing recommendation rationale"
