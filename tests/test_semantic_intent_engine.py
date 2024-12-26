import pytest
import numpy as np

from src.semantic_understanding.semantic_intent_engine import (
    SemanticIntentEngine,
    SemanticContext
)

class TestSemanticIntentEngine:
    @pytest.fixture
    def semantic_intent_engine(self):
        """Create a semantic intent engine for testing"""
        return SemanticIntentEngine()
    
    def test_text_preprocessing(self, semantic_intent_engine):
        """
        Test text preprocessing functionality
        
        Validates:
        - Tokenization
        - Stopword removal
        - Text normalization
        """
        # Test cases with different input texts
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Quantum computing is a revolutionary technology!",
            "How can we improve machine learning algorithms?"
        ]
        
        for text in test_texts:
            # Preprocess text
            preprocessed_text = semantic_intent_engine.preprocess_text(text)
            
            # Validate preprocessing
            assert isinstance(preprocessed_text, str), "Preprocessing should return a string"
            
            # Check basic preprocessing properties
            preprocessed_tokens = preprocessed_text.split()
            assert len(preprocessed_tokens) > 0, "Preprocessing should not result in empty text"
            
            # Validate stopword removal
            stop_words = {'the', 'a', 'an', 'over'}
            assert not any(token.lower() in stop_words for token in preprocessed_tokens), "Stopwords should be removed"
    
    def test_semantic_feature_extraction(self, semantic_intent_engine):
        """
        Test semantic feature extraction
        
        Validates:
        - Feature extraction process
        - Domain classification
        - Emotion detection
        """
        # Test texts from different domains
        test_texts = [
            "Quantum entanglement in quantum computing research",
            "Machine learning algorithms for predictive analytics",
            "Advanced engineering design principles"
        ]
        
        for text in test_texts:
            # Extract semantic features
            features = semantic_intent_engine.extract_semantic_features(text)
            
            # Validate feature extraction
            assert 'domains' in features, "Missing domain classification"
            assert 'emotion' in features, "Missing emotion detection"
            assert 'language' in features, "Missing language detection"
            assert 'features' in features, "Missing feature vector"
            
            # Validate domain classification
            domains = features['domains']
            assert isinstance(domains, list), "Domain classification should return a list"
            assert len(domains) > 0, "At least one domain should be classified"
            
            # Check domain probabilities
            for domain, probability in domains:
                assert domain in semantic_intent_engine.domains, f"Invalid domain: {domain}"
                assert 0 <= probability <= 1, f"Invalid probability for domain {domain}"
            
            # Validate emotion detection
            emotion = features['emotion']
            assert emotion in ['positive', 'negative', 'neutral'], f"Invalid emotion: {emotion}"
            
            # Validate feature vector
            feature_vector = features['features']
            assert isinstance(feature_vector, np.ndarray), "Feature vector should be a numpy array"
            assert feature_vector.size > 0, "Feature vector should not be empty"
    
    def test_semantic_context_generation(self, semantic_intent_engine):
        """
        Test semantic context generation
        
        Validates:
        - Context generation process
        - Personalization
        - Metadata inclusion
        """
        # Test texts
        test_texts = [
            "Explore quantum computing research methodologies",
            "Develop advanced machine learning algorithms"
        ]
        
        # Test user profiles
        user_profiles = [
            None,
            {
                'preferred_domains': ['technology', 'science'],
                'explanation_level': 'detailed'
            }
        ]
        
        for text in test_texts:
            for user_profile in user_profiles:
                # Generate semantic context
                semantic_context = semantic_intent_engine.generate_semantic_context(
                    text, 
                    user_profile
                )
                
                # Validate semantic context
                assert isinstance(semantic_context, SemanticContext), "Invalid semantic context type"
                
                # Check context properties
                assert semantic_context.domain is not None, "Domain should be set"
                assert semantic_context.intent is not None, "Intent should be set"
                assert semantic_context.emotion is not None, "Emotion should be set"
                assert semantic_context.language is not None, "Language should be set"
                assert 0 <= semantic_context.confidence <= 1, "Invalid confidence score"
                
                # Validate metadata
                assert 'domains' in semantic_context.metadata, "Missing domains in metadata"
                assert 'features' in semantic_context.metadata, "Missing features in metadata"
                
                # Check personalization
                if user_profile:
                    if 'preferred_domains' in user_profile:
                        assert semantic_context.domain in user_profile['preferred_domains'], "Domain not personalized"
                    
                    if 'explanation_level' in user_profile:
                        assert 'explanation_granularity' in semantic_context.metadata, "Explanation granularity not set"
    
    def test_intent_inference(self, semantic_intent_engine):
        """
        Test intent inference mechanism
        
        Validates:
        - Intent classification
        - Keyword-based intent detection
        """
        # Test texts with different intents
        test_texts = [
            "What is quantum computing?",
            "Provide an analysis of machine learning trends",
            "Execute the research algorithm",
            "Investigate the latest technological advancements"
        ]
        
        expected_intents = ['query', 'request', 'command', 'explore']
        
        for text, expected_intent in zip(test_texts, expected_intents):
            # Infer intent
            intent = semantic_intent_engine._infer_intent(text)
            
            # Validate intent
            assert intent == expected_intent, f"Incorrect intent for text: {text}"
    
    def test_semantic_similarity(self, semantic_intent_engine):
        """
        Test semantic similarity computation
        
        Validates:
        - Similarity calculation
        - Similarity score range
        """
        # Test text pairs
        test_pairs = [
            (
                "Quantum computing is a revolutionary technology",
                "Advanced quantum computing research methods"
            ),
            (
                "Machine learning algorithms for predictive analytics",
                "Data science and machine learning techniques"
            ),
            (
                "Completely unrelated text about random topics",
                "Another completely different text"
            )
        ]
        
        for text1, text2 in test_pairs:
            # Compute semantic similarity
            similarity = semantic_intent_engine.compute_semantic_similarity(text1, text2)
            
            # Validate similarity
            assert isinstance(similarity, float), "Similarity should be a float"
            assert 0 <= similarity <= 1, f"Invalid similarity score for texts: {text1}, {text2}"
    
    def test_emotion_detection(self, semantic_intent_engine):
        """
        Test emotion detection mechanism
        
        Validates:
        - Emotion classification
        - Lexicon-based emotion detection
        """
        # Test texts with different emotional tones
        test_texts = [
            "I'm excited about the latest quantum computing breakthrough!",
            "The research results are disappointing and confusing.",
            "Let's analyze the data objectively and systematically."
        ]
        
        expected_emotions = ['positive', 'negative', 'neutral']
        
        for text, expected_emotion in zip(test_texts, expected_emotions):
            # Detect emotion
            emotion = semantic_intent_engine._detect_emotion(
                semantic_intent_engine.preprocess_text(text)
            )
            
            # Validate emotion
            assert emotion == expected_emotion, f"Incorrect emotion for text: {text}"
