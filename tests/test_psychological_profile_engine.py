import pytest
from typing import Dict, List, Any
import numpy as np

from src.interaction.psychological_profile_engine import (
    PsychologicalProfileEngine,
    CognitiveStyle,
    PersonalityProfile,
    LearningPreference,
    UserProfile
)

class TestPsychologicalProfileEngine:
    @pytest.fixture
    def profile_engine(self):
        """Create profile engine instance for testing"""
        return PsychologicalProfileEngine()
    
    @pytest.fixture
    def sample_assessment(self) -> Dict[str, float]:
        """Create sample psychological assessment data"""
        return {
            'analytical': 0.8,
            'intuitive': 0.6,
            'visual': 0.7,
            'verbal': 0.5,
            'sequential': 0.9,
            'global': 0.4,
            'active': 0.6,
            'reflective': 0.7,
            'openness': 0.8,
            'conscientiousness': 0.7,
            'extraversion': 0.6,
            'agreeableness': 0.8,
            'neuroticism': 0.3,
            'visual_learning': 0.7,
            'auditory_learning': 0.6,
            'kinesthetic_learning': 0.8,
            'reading_learning': 0.7,
            'multimodal_learning': 0.6
        }
    
    @pytest.fixture
    def sample_interaction_data(self) -> Dict[str, Any]:
        """Create sample interaction data"""
        return {
            'interaction_type': 'query',
            'content': 'Complex technical question about quantum mechanics',
            'duration': 120,
            'engagement_metrics': {
                'attention_score': 0.8,
                'comprehension_score': 0.7,
                'interaction_depth': 0.9
            },
            'response_metrics': {
                'satisfaction_score': 0.8,
                'clarity_rating': 0.9,
                'helpfulness_rating': 0.85
            }
        }
    
    def test_profile_creation(self, profile_engine, sample_assessment):
        """Test user profile creation"""
        user_id = 'test_user_1'
        profile = profile_engine.create_user_profile(
            user_id=user_id,
            initial_assessment=sample_assessment
        )
        
        # Verify profile structure
        assert isinstance(profile, UserProfile)
        assert profile.user_id == user_id
        assert isinstance(profile.cognitive_style, CognitiveStyle)
        assert isinstance(profile.personality_profile, PersonalityProfile)
        assert isinstance(profile.learning_preference, LearningPreference)
        
        # Verify cognitive style scores
        assert 0 <= profile.cognitive_style.analytical_score <= 1
        assert 0 <= profile.cognitive_style.intuitive_score <= 1
        assert 0 <= profile.cognitive_style.visual_score <= 1
        assert 0 <= profile.cognitive_style.verbal_score <= 1
        
        # Verify personality traits
        assert 0 <= profile.personality_profile.openness <= 1
        assert 0 <= profile.personality_profile.conscientiousness <= 1
        assert 0 <= profile.personality_profile.extraversion <= 1
        assert 0 <= profile.personality_profile.agreeableness <= 1
        assert 0 <= profile.personality_profile.neuroticism <= 1
        
        # Verify learning preferences
        assert 0 <= profile.learning_preference.visual_preference <= 1
        assert 0 <= profile.learning_preference.auditory_preference <= 1
        assert 0 <= profile.learning_preference.kinesthetic_preference <= 1
        assert 0 <= profile.learning_preference.reading_preference <= 1
    
    def test_profile_update(self, profile_engine, sample_assessment, sample_interaction_data):
        """Test profile updates based on interaction data"""
        # Create initial profile
        profile = profile_engine.create_user_profile(
            user_id='test_user_2',
            initial_assessment=sample_assessment
        )
        
        # Store initial scores
        initial_analytical = profile.cognitive_style.analytical_score
        initial_visual = profile.learning_preference.visual_preference
        
        # Update profile
        updated_profile = profile_engine.update_profile(
            profile=profile,
            interaction_data=sample_interaction_data
        )
        
        # Verify updates
        assert len(updated_profile.interaction_history) == 1
        assert updated_profile.interaction_history[0] == sample_interaction_data
        assert len(updated_profile.adaptation_metrics) > 0
    
    def test_interaction_strategy_generation(self, profile_engine, sample_assessment):
        """Test interaction strategy generation"""
        # Create profile
        profile = profile_engine.create_user_profile(
            user_id='test_user_3',
            initial_assessment=sample_assessment
        )
        
        # Generate strategy
        context = {
            'task_type': 'technical_explanation',
            'complexity_level': 'high',
            'time_constraint': 'medium'
        }
        
        strategy = profile_engine.generate_interaction_strategy(
            profile=profile,
            context=context
        )
        
        # Verify strategy
        assert isinstance(strategy, dict)
        assert 'content_type' in strategy
        assert 'interaction_mode' in strategy
        assert 'presentation_style' in strategy
    
    def test_interaction_pattern_analysis(self, profile_engine, sample_assessment, sample_interaction_data):
        """Test interaction pattern analysis"""
        # Create profile with interaction history
        profile = profile_engine.create_user_profile(
            user_id='test_user_4',
            initial_assessment=sample_assessment
        )
        
        # Add multiple interactions
        for _ in range(profile_engine.profile_config['min_interaction_history']):
            profile = profile_engine.update_profile(
                profile=profile,
                interaction_data=sample_interaction_data
            )
        
        # Analyze patterns
        analysis = profile_engine.analyze_interaction_patterns(profile)
        
        # Verify analysis
        assert isinstance(analysis, dict)
        assert 'dominant_patterns' in analysis
        assert 'style_evolution' in analysis
        assert 'adaptation_recommendations' in analysis
    
    def test_cognitive_indicators(self, profile_engine, sample_interaction_data):
        """Test cognitive indicator extraction"""
        indicators = profile_engine._extract_cognitive_indicators(sample_interaction_data)
        
        # Verify indicators
        assert isinstance(indicators, dict)
        assert 'analytical' in indicators
        assert 'intuitive' in indicators
        assert 'visual' in indicators
        assert 'verbal' in indicators
        
        # Verify indicator values
        for value in indicators.values():
            assert 0 <= value <= 1
    
    def test_learning_indicators(self, profile_engine, sample_interaction_data):
        """Test learning indicator extraction"""
        indicators = profile_engine._extract_learning_indicators(sample_interaction_data)
        
        # Verify indicators
        assert isinstance(indicators, dict)
        assert 'visual' in indicators
        assert 'auditory' in indicators
        assert 'kinesthetic' in indicators
        assert 'reading' in indicators
        
        # Verify indicator values
        for value in indicators.values():
            assert 0 <= value <= 1
    
    def test_adaptation_metrics(self, profile_engine, sample_assessment):
        """Test adaptation metrics computation"""
        # Create profile
        profile = profile_engine.create_user_profile(
            user_id='test_user_5',
            initial_assessment=sample_assessment
        )
        
        # Compute metrics
        metrics = profile_engine._compute_adaptation_metrics(profile)
        
        # Verify metrics
        assert isinstance(metrics, dict)
        assert 'style_stability' in metrics
        assert 'learning_efficiency' in metrics
        assert 'interaction_satisfaction' in metrics
        
        # Verify metric values
        for value in metrics.values():
            assert 0 <= value <= 1
    
    def test_personalization_consistency(self, profile_engine, sample_assessment):
        """Test consistency of personalization strategies"""
        # Create two identical profiles
        profile1 = profile_engine.create_user_profile(
            user_id='test_user_6a',
            initial_assessment=sample_assessment
        )
        
        profile2 = profile_engine.create_user_profile(
            user_id='test_user_6b',
            initial_assessment=sample_assessment
        )
        
        # Generate strategies with same context
        context = {
            'task_type': 'technical_explanation',
            'complexity_level': 'high'
        }
        
        strategy1 = profile_engine.generate_interaction_strategy(profile1, context)
        strategy2 = profile_engine.generate_interaction_strategy(profile2, context)
        
        # Verify strategies are identical
        assert strategy1 == strategy2
