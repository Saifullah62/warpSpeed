import pytest
import json
import torch

from src.interface.natural_language_interface import (
    NaturalLanguageInterface,
    MultilingualQueryProcessor,
    DialogueState,
    ExplanationGenerator,
    initialize_natural_language_interface
)
from src.interface.visualization_system import (
    KnowledgeGraphVisualizer,
    TechnologyDependencyVisualizer,
    ResearchProgressTracker,
    initialize_visualization_system
)
from src.knowledge_graph.schema import Entity, EntityType
from src.semantic_understanding.semantic_understanding_engine import SemanticUnderstandingEngine
from src.personalized_interaction.personalized_interaction_model import PersonalizedInteractionModel

class TestMultilingualQueryProcessor:
    @pytest.fixture
    def query_processor(self):
        """Create a multilingual query processor."""
        return MultilingualQueryProcessor()
    
    def test_language_detection(self, query_processor):
        """
        Test language detection capabilities.
        
        Validates:
        - Accurate language detection
        - Fallback to English
        """
        # Test various language inputs
        test_queries = [
            ("Hello, how are you?", 'en'),
            ("Bonjour, comment ça va?", 'fr'),
            ("Hola, ¿cómo estás?", 'es')
        ]
        
        for query, expected_lang in test_queries:
            detected_lang = query_processor.detect_language(query)
            assert detected_lang == expected_lang, f"Failed to detect {expected_lang}"
    
    def test_query_translation(self, query_processor):
        """
        Test query translation mechanism.
        
        Validates:
        - Successful translation
        - Preserving query meaning
        """
        # Test translation scenarios
        test_queries = [
            ("Bonjour, comment fonctionne l'intelligence artificielle?", 'en'),
            ("¿Cuáles son los avances en computación cuántica?", 'en')
        ]
        
        for query, target_lang in test_queries:
            translated_query = query_processor.translate_query(query, target_lang)
            
            assert isinstance(translated_query, str), "Translation failed"
            assert len(translated_query) > 0, "Empty translation"

class TestDialogueState:
    @pytest.fixture
    def dialogue_state(self):
        """Create a dialogue state for testing."""
        return DialogueState(user_id="test_user")
    
    def test_context_update(self, dialogue_state):
        """
        Test dialogue context update mechanism.
        
        Validates:
        - Context history tracking
        - Entity extraction
        - Context size management
        """
        # Simulate multiple interactions
        test_interactions = [
            {
                'query': "Tell me about quantum computing",
                'response': "Quantum computing is a revolutionary technology...",
                'entities': [
                    Entity(name="Quantum Computing", type=EntityType.TECHNOLOGY)
                ]
            },
            {
                'query': "What are its applications?",
                'response': "Quantum computing has applications in cryptography, drug discovery...",
                'entities': [
                    Entity(name="Cryptography", type=EntityType.CONCEPT),
                    Entity(name="Drug Discovery", type=EntityType.CONCEPT)
                ]
            }
        ]
        
        for interaction in test_interactions:
            dialogue_state.update_context(
                interaction['query'], 
                interaction['response'], 
                interaction['entities']
            )
        
        # Validate context tracking
        assert len(dialogue_state.conversation_history) > 0, "Conversation history not updated"
        assert len(dialogue_state.context_entities) > 0, "Context entities not tracked"
        
        # Check context size limits
        assert len(dialogue_state.conversation_history) <= 10, "Exceeded conversation history limit"
        assert len(dialogue_state.context_entities) <= 20, "Exceeded context entities limit"

class TestNaturalLanguageInterface:
    @pytest.fixture
    def natural_language_interface(self):
        """Create a natural language interface for testing."""
        return initialize_natural_language_interface()[0]
    
    def test_query_processing(self, natural_language_interface):
        """
        Test comprehensive query processing.
        
        Validates:
        - Query understanding
        - Entity extraction
        - Reasoning integration
        """
        test_queries = [
            "What are the latest advancements in quantum computing?",
            "Explain the relationship between AI and machine learning"
        ]
        
        for query in test_queries:
            result = natural_language_interface.process_query(
                query, 
                user_id="test_user"
            )
            
            # Validate result structure
            assert 'response' in result, "Missing response"
            assert 'entities' in result, "Missing entities"
            assert 'reasoning_confidence' in result, "Missing reasoning confidence"
            assert 'language' in result, "Missing language"
            
            # Check result quality
            assert len(result['response']) > 0, "Empty response"
            assert result['reasoning_confidence'] >= 0, "Invalid reasoning confidence"

class TestExplanationGenerator:
    @pytest.fixture
    def explanation_generator(self):
        """Create an explanation generator for testing."""
        _, explanation_generator = initialize_natural_language_interface()
        return explanation_generator
    
    def test_reasoning_explanation(self, explanation_generator):
        """
        Test reasoning explanation generation.
        
        Validates:
        - Explanation generation
        - Comprehensibility
        """
        # Simulate reasoning result
        reasoning_result = {
            'entities': ['Quantum Computing', 'Machine Learning'],
            'relationships': [('Quantum Computing', 'Machine Learning', 'ENABLES')],
            'confidence': 0.85
        }
        
        explanation = explanation_generator.generate_reasoning_explanation(reasoning_result)
        
        # Validate explanation
        assert isinstance(explanation, str), "Invalid explanation type"
        assert len(explanation) > 0, "Empty explanation"

class TestKnowledgeGraphVisualizer:
    @pytest.fixture
    def knowledge_graph_visualizer(self):
        """Create a knowledge graph visualizer for testing."""
        return initialize_visualization_system()[0]
    
    def test_graph_visualization(self, knowledge_graph_visualizer):
        """
        Test knowledge graph visualization generation.
        
        Validates:
        - Visualization creation
        - Node and edge handling
        """
        # Test visualization generation
        visualization_result = knowledge_graph_visualizer.generate_graph_visualization(
            max_nodes=50,
            node_type=EntityType.TECHNOLOGY
        )
        
        # Validate visualization result
        assert 'visualization' in visualization_result, "Missing visualization data"
        assert 'node_count' in visualization_result, "Missing node count"
        assert 'edge_count' in visualization_result, "Missing edge count"
        
        # Check visualization data
        visualization_data = json.loads(visualization_result['visualization'])
        assert 'data' in visualization_data, "Invalid visualization structure"
        assert len(visualization_data['data']) > 0, "Empty visualization data"

class TestTechnologyDependencyVisualizer:
    @pytest.fixture
    def technology_dependency_visualizer(self):
        """Create a technology dependency visualizer for testing."""
        return initialize_visualization_system()[1]
    
    def test_dependency_tree_generation(self, technology_dependency_visualizer):
        """
        Test technology dependency tree visualization.
        
        Validates:
        - Dependency tree creation
        - Depth handling
        """
        # Test dependency tree generation
        dependency_result = technology_dependency_visualizer.generate_dependency_tree(
            root_technology="Quantum Computing",
            depth=3
        )
        
        # Validate dependency result
        assert 'visualization' in dependency_result, "Missing visualization data"
        assert 'dependency_depth' in dependency_result, "Missing dependency depth"
        assert 'total_dependencies' in dependency_result, "Missing total dependencies"
        
        # Check visualization data
        visualization_data = json.loads(dependency_result['visualization'])
        assert 'data' in visualization_data, "Invalid visualization structure"
        assert len(visualization_data['data']) > 0, "Empty visualization data"

class TestResearchProgressTracker:
    @pytest.fixture
    def research_progress_tracker(self):
        """Create a research progress tracker for testing."""
        return initialize_visualization_system()[2]
    
    def test_progress_dashboard_generation(self, research_progress_tracker):
        """
        Test research progress dashboard generation.
        
        Validates:
        - Dashboard creation
        - Domain tracking
        """
        # Test progress dashboard generation
        research_domains = [
            "Quantum Computing", 
            "Artificial Intelligence", 
            "Biotechnology"
        ]
        
        progress_result = research_progress_tracker.generate_progress_dashboard(
            research_domains
        )
        
        # Validate progress result
        assert 'visualization' in progress_result, "Missing visualization data"
        assert 'domains_tracked' in progress_result, "Missing domains tracked"
        
        # Check visualization data
        visualization_data = json.loads(progress_result['visualization'])
        assert 'data' in visualization_data, "Invalid visualization structure"
        assert len(visualization_data['data']) > 0, "Empty visualization data"
        assert progress_result['domains_tracked'] == len(research_domains), "Incorrect domain tracking"

class TestSemanticUnderstandingEngine:
    @pytest.fixture
    def semantic_understanding_engine(self, knowledge_graph, embedding_model):
        """Create a semantic understanding engine for testing."""
        return SemanticUnderstandingEngine(knowledge_graph, embedding_model)
    
    def test_semantic_intent_extraction(self, semantic_understanding_engine):
        """
        Test semantic intent extraction mechanism.
        
        Validates:
        - Domain classification
        - Semantic embedding generation
        - Contextual analysis
        """
        test_queries = [
            "What are the latest advancements in quantum computing?",
            "Explain the principles of machine learning",
            "How do biological systems process information?"
        ]
        
        for query in test_queries:
            intent_result = semantic_understanding_engine.extract_semantic_intent(query)
            
            # Validate intent result structure
            assert 'domains' in intent_result, "Missing domains in semantic intent"
            assert 'domain_scores' in intent_result, "Missing domain scores"
            assert 'semantic_embedding' in intent_result, "Missing semantic embedding"
            assert 'contextual_boost' in intent_result, "Missing contextual boost"
            
            # Check domain classification
            assert len(intent_result['domains']) > 0, "No domains classified"
            assert all(0 <= score <= 1 for score in intent_result['domain_scores']), "Invalid domain scores"
            
            # Check semantic embedding
            assert intent_result['semantic_embedding'] is not None, "Semantic embedding not generated"
    
    def test_context_analysis(self, semantic_understanding_engine):
        """
        Test context analysis and semantic boosting.
        
        Validates:
        - Context-aware semantic analysis
        - Contextual domain relevance
        """
        # Create mock dialogue state
        mock_dialogue_state = DialogueState(user_id="test_user")
        mock_dialogue_state.conversation_history = [
            {
                'query': "Tell me about quantum computing",
                'response': "Quantum computing uses quantum mechanics principles...",
                'timestamp': datetime.now().isoformat()
            },
            {
                'query': "What are quantum algorithms?",
                'response': "Quantum algorithms leverage quantum superposition...",
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        # Analyze context
        context_boost = semantic_understanding_engine._analyze_context(mock_dialogue_state)
        
        # Validate context boost
        assert isinstance(context_boost, dict), "Invalid context boost type"
        assert len(context_boost) > 0, "Empty context boost"
        
        # Check domain relevance scores
        for domain, score in context_boost.items():
            assert 0 <= score <= 1, f"Invalid score for domain {domain}"

class TestPersonalizedInteractionModel:
    @pytest.fixture
    def personalized_interaction_model(self, semantic_understanding_engine):
        """Create a personalized interaction model for testing."""
        return PersonalizedInteractionModel(semantic_understanding_engine)
    
    def test_user_preference_learning(self, personalized_interaction_model):
        """
        Test user preference learning mechanism.
        
        Validates:
        - User profile creation
        - Domain interest tracking
        - Interaction style adaptation
        """
        user_id = "test_user"
        interaction_history = [
            {
                'query': "Explain quantum computing principles",
                'response': "Quantum computing leverages quantum mechanics...",
                'timestamp': datetime.now().isoformat()
            },
            {
                'query': "What are the latest AI research trends?",
                'response': "AI research is focusing on transformer models...",
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        # Learn user preferences
        personalized_interaction_model.learn_user_preferences(
            user_id, 
            interaction_history
        )
        
        # Validate user profile
        assert user_id in personalized_interaction_model.user_profiles, "User profile not created"
        
        profile = personalized_interaction_model.user_profiles[user_id]
        
        # Check profile attributes
        assert 'interaction_style' in profile, "Missing interaction style"
        assert 'domain_interests' in profile, "Missing domain interests"
        assert 'interaction_count' in profile, "Missing interaction count"
        
        # Validate domain interests
        assert len(profile['domain_interests']) > 0, "No domain interests tracked"
        
        # Check interaction style
        assert profile['interaction_style'] in ['technical', 'casual', 'balanced'], "Invalid interaction style"
    
    def test_personalized_interaction_config(self, personalized_interaction_model):
        """
        Test personalized interaction configuration retrieval.
        
        Validates:
        - Configuration generation
        - Style-based configuration
        """
        user_id = "test_user"
        
        # Simulate user profile creation
        personalized_interaction_model.user_profiles[user_id] = {
            'interaction_style': 'technical',
            'domain_interests': {'technology': 0.9, 'science': 0.7},
            'interaction_count': 10
        }
        
        # Retrieve personalized configuration
        config = personalized_interaction_model.get_personalized_interaction_config(user_id)
        
        # Validate configuration
        assert isinstance(config, dict), "Invalid configuration type"
        
        # Check configuration keys
        expected_keys = ['verbosity', 'explanation_depth', 'visualization_preference']
        for key in expected_keys:
            assert key in config, f"Missing {key} in interaction configuration"
        
        # Validate technical style configuration
        assert config['verbosity'] == 'high', "Incorrect verbosity for technical style"
        assert config['explanation_depth'] == 'detailed', "Incorrect explanation depth for technical style"
        assert config['visualization_preference'] == 'graph', "Incorrect visualization preference for technical style"

class TestNaturalLanguageInterface:
    @pytest.fixture
    def natural_language_interface(self):
        """Create a natural language interface for testing."""
        return initialize_natural_language_interface()[0]
    
    def test_enhanced_query_processing(self, natural_language_interface):
        """
        Test enhanced query processing with semantic understanding and personalization.
        
        Validates:
        - Semantic intent extraction
        - Personalized interaction configuration
        - Comprehensive query result
        """
        test_queries = [
            "What are the latest advancements in quantum computing?",
            "Explain machine learning algorithms"
        ]
        
        for query in test_queries:
            result = natural_language_interface.process_query(
                query, 
                user_id="test_user"
            )
            
            # Validate enhanced result structure
            assert 'semantic_intent' in result, "Missing semantic intent"
            assert 'interaction_config' in result, "Missing interaction configuration"
            
            # Check semantic intent
            semantic_intent = result['semantic_intent']
            assert 'domains' in semantic_intent, "Missing domains in semantic intent"
            assert 'domain_scores' in semantic_intent, "Missing domain scores"
            
            # Check interaction configuration
            interaction_config = result['interaction_config']
            assert 'verbosity' in interaction_config, "Missing verbosity in interaction config"
            assert 'explanation_depth' in interaction_config, "Missing explanation depth"
            assert 'visualization_preference' in interaction_config, "Missing visualization preference"

def test_interface_system_initialization():
    """
    Test complete interface system initialization.
    
    Validates:
    - System components creation
    - Correct component types
    """
    # Test natural language interface initialization
    natural_language_interface, explanation_generator = initialize_natural_language_interface()
    
    assert natural_language_interface is not None, "Natural language interface not initialized"
    assert explanation_generator is not None, "Explanation generator not initialized"
    
    # Test visualization system initialization
    knowledge_graph_visualizer, technology_dependency_visualizer, research_progress_tracker = initialize_visualization_system()
    
    assert knowledge_graph_visualizer is not None, "Knowledge graph visualizer not initialized"
    assert technology_dependency_visualizer is not None, "Technology dependency visualizer not initialized"
    assert research_progress_tracker is not None, "Research progress tracker not initialized"
