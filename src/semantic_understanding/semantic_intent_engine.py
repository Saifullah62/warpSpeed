import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

@dataclass
class SemanticContext:
    """
    Represents a comprehensive semantic context for an interaction
    """
    domain: str = 'generic'
    intent: str = 'undefined'
    emotion: str = 'neutral'
    language: str = 'en'
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class SemanticIntentEngine:
    def __init__(
        self, 
        domains: Optional[List[str]] = None,
        languages: Optional[List[str]] = None
    ):
        """
        Initialize Semantic Intent Engine
        
        Args:
            domains: Predefined domains for classification
            languages: Supported languages for semantic analysis
        """
        # Download necessary NLTK resources
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # Domain and language configuration
        self.domains = domains or [
            'technology', 'science', 'research', 
            'engineering', 'mathematics', 'computing'
        ]
        
        self.languages = languages or ['en', 'es', 'fr']
        
        # Feature extraction configurations
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        # Multi-label classification
        self.domain_classifier = OneVsRestClassifier(
            LinearSVC(random_state=42, max_iter=5000)
        )
        
        # Emotion and sentiment analysis
        self.emotion_lexicon = {
            'positive': ['excited', 'happy', 'optimistic', 'enthusiastic'],
            'negative': ['frustrated', 'confused', 'disappointed', 'uncertain'],
            'neutral': ['calm', 'objective', 'analytical']
        }
        
        # Language detection configuration
        self.language_detection_threshold = 0.7
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess input text for semantic analysis
        
        Args:
            text: Input text to preprocess
        
        Returns:
            Preprocessed text
        """
        # Tokenize and remove stopwords
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        
        preprocessed_tokens = [
            token for token in tokens 
            if token.isalnum() and token not in stop_words
        ]
        
        return ' '.join(preprocessed_tokens)
    
    def extract_semantic_features(
        self, 
        text: str
    ) -> Dict[str, Any]:
        """
        Extract comprehensive semantic features
        
        Args:
            text: Input text for feature extraction
        
        Returns:
            Extracted semantic features
        """
        # Preprocess text
        preprocessed_text = self.preprocess_text(text)
        
        # TF-IDF feature extraction
        features = self.vectorizer.transform([preprocessed_text])
        
        # Domain classification
        domain_probabilities = self.domain_classifier.predict_proba(features)
        top_domains = sorted(
            zip(self.domains, domain_probabilities[0]), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        # Emotion detection
        emotion = self._detect_emotion(preprocessed_text)
        
        # Language detection
        language = self._detect_language(text)
        
        return {
            'domains': top_domains,
            'emotion': emotion,
            'language': language,
            'features': features.toarray()[0]
        }
    
    def _detect_emotion(self, text: str) -> str:
        """
        Detect emotional tone of the text
        
        Args:
            text: Preprocessed text
        
        Returns:
            Detected emotion
        """
        # Simple lexicon-based emotion detection
        tokens = set(text.split())
        
        emotion_scores = {
            'positive': sum(1 for word in tokens if word in self.emotion_lexicon['positive']),
            'negative': sum(1 for word in tokens if word in self.emotion_lexicon['negative']),
            'neutral': sum(1 for word in tokens if word in self.emotion_lexicon['neutral'])
        }
        
        return max(emotion_scores, key=emotion_scores.get)
    
    def _detect_language(self, text: str) -> str:
        """
        Detect language of the text
        
        Args:
            text: Input text
        
        Returns:
            Detected language code
        """
        # Placeholder for language detection
        # In a production system, use libraries like langdetect
        return 'en'
    
    def generate_semantic_context(
        self, 
        text: str, 
        user_profile: Optional[Dict[str, Any]] = None
    ) -> SemanticContext:
        """
        Generate comprehensive semantic context
        
        Args:
            text: Input text
            user_profile: Optional user profile for personalization
        
        Returns:
            Semantic context object
        """
        # Extract semantic features
        features = self.extract_semantic_features(text)
        
        # Create semantic context
        semantic_context = SemanticContext(
            domain=features['domains'][0][0],
            intent=self._infer_intent(text),
            emotion=features['emotion'],
            language=features['language'],
            confidence=features['domains'][0][1],
            metadata={
                'domains': features['domains'],
                'features': features['features']
            }
        )
        
        # Personalize context if user profile is provided
        if user_profile:
            semantic_context = self._personalize_context(
                semantic_context, 
                user_profile
            )
        
        return semantic_context
    
    def _infer_intent(self, text: str) -> str:
        """
        Infer intent from text
        
        Args:
            text: Input text
        
        Returns:
            Inferred intent
        """
        # Simple intent inference based on keywords
        intent_keywords = {
            'query': ['what', 'how', 'why', 'explain', 'describe'],
            'request': ['give', 'provide', 'show', 'help'],
            'command': ['do', 'execute', 'run', 'perform'],
            'explore': ['investigate', 'research', 'analyze', 'study']
        }
        
        tokens = set(text.lower().split())
        
        for intent, keywords in intent_keywords.items():
            if any(keyword in tokens for keyword in keywords):
                return intent
        
        return 'undefined'
    
    def _personalize_context(
        self, 
        semantic_context: SemanticContext, 
        user_profile: Dict[str, Any]
    ) -> SemanticContext:
        """
        Personalize semantic context based on user profile
        
        Args:
            semantic_context: Original semantic context
            user_profile: User profile information
        
        Returns:
            Personalized semantic context
        """
        # Adjust context based on user preferences
        if 'preferred_domains' in user_profile:
            preferred_domains = user_profile['preferred_domains']
            if semantic_context.domain not in preferred_domains:
                # Adjust domain if not in preferred list
                semantic_context.domain = preferred_domains[0]
        
        # Adjust explanation granularity
        if 'explanation_level' in user_profile:
            semantic_context.metadata['explanation_granularity'] = user_profile['explanation_level']
        
        return semantic_context
    
    def compute_semantic_similarity(
        self, 
        text1: str, 
        text2: str
    ) -> float:
        """
        Compute semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Semantic similarity score
        """
        # Preprocess texts
        preprocessed_text1 = self.preprocess_text(text1)
        preprocessed_text2 = self.preprocess_text(text2)
        
        # Compute TF-IDF vectors
        vectors = self.vectorizer.transform([preprocessed_text1, preprocessed_text2])
        
        # Compute cosine similarity
        similarity = np.dot(vectors[0], vectors[1]) / (
            np.linalg.norm(vectors[0]) * np.linalg.norm(vectors[1])
        )
        
        return float(similarity)
