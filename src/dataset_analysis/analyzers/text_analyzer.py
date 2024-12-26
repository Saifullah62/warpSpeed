from typing import Dict, Any, List, Optional
from datasets import Dataset
import numpy as np
from collections import Counter
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import torch
from .base_analyzer import BaseAnalyzer

class TextAnalyzer(BaseAnalyzer):
    """Analyzer for text datasets."""
    
    def __init__(self, 
                 text_column: str = 'text',
                 sentiment_model: str = 'distilbert-base-uncased-finetuned-sst-2-english',
                 n_topics: int = 5):
        super().__init__()
        self.text_column = text_column
        self.sentiment_analyzer = pipeline('sentiment-analysis', 
                                         model=sentiment_model,
                                         device=0 if torch.cuda.is_available() else -1)
        self.n_topics = n_topics
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.topic_model = LatentDirichletAllocation(n_components=n_topics)
        
    def analyze(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Analyze text dataset.
        
        Args:
            dataset: Dataset containing text data
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Basic text statistics
            text_lengths = [len(text.split()) for text in dataset[self.text_column]]
            vocab = set(word for text in dataset[self.text_column] 
                       for word in text.lower().split())
            
            # Sentiment analysis
            sentiments = self.sentiment_analyzer(dataset[self.text_column][:100])
            sentiment_distribution = Counter(s['label'] for s in sentiments)
            
            # Topic modeling
            tfidf_matrix = self.vectorizer.fit_transform(dataset[self.text_column])
            topics = self.topic_model.fit_transform(tfidf_matrix)
            
            # Feature names for topics
            feature_names = self.vectorizer.get_feature_names_out()
            top_words_per_topic = []
            for topic_idx, topic in enumerate(self.topic_model.components_):
                top_words = [feature_names[i] 
                           for i in topic.argsort()[:-10:-1]]
                top_words_per_topic.append(top_words)
            
            metrics = {
                'avg_text_length': np.mean(text_lengths),
                'std_text_length': np.std(text_lengths),
                'vocabulary_size': len(vocab),
                'sentiment_distribution': dict(sentiment_distribution),
                'topics': top_words_per_topic
            }
            
            self.track_metrics(metrics)
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error in text analysis: {str(e)}")
            raise
            
    def detect_anomalies(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Detect anomalies in text dataset.
        
        Args:
            dataset: Dataset containing text data
            
        Returns:
            Dictionary containing detected anomalies
        """
        anomalies = {}
        
        try:
            # Check for extremely short or long texts
            text_lengths = [len(text.split()) for text in dataset[self.text_column]]
            mean_length = np.mean(text_lengths)
            std_length = np.std(text_lengths)
            
            length_anomalies = []
            for idx, length in enumerate(text_lengths):
                if abs(length - mean_length) > 3 * std_length:
                    length_anomalies.append({
                        'index': idx,
                        'length': length,
                        'z_score': (length - mean_length) / std_length
                    })
            
            # Check for unusual character distributions
            char_distributions = []
            for text in dataset[self.text_column]:
                chars = Counter(text.lower())
                total = sum(chars.values())
                distribution = {char: count/total 
                              for char, count in chars.items()}
                char_distributions.append(distribution)
            
            # Calculate average distribution
            avg_distribution = {}
            for dist in char_distributions:
                for char, freq in dist.items():
                    avg_distribution[char] = avg_distribution.get(char, 0) + freq
            avg_distribution = {char: freq/len(char_distributions) 
                              for char, freq in avg_distribution.items()}
            
            # Find texts with unusual character distributions
            char_anomalies = []
            for idx, dist in enumerate(char_distributions):
                diff = sum(abs(dist.get(char, 0) - avg_distribution.get(char, 0))
                          for char in set(dist) | set(avg_distribution))
                if diff > 0.5:  # Threshold for character distribution difference
                    char_anomalies.append({
                        'index': idx,
                        'difference_score': diff
                    })
            
            anomalies = {
                'length_anomalies': length_anomalies,
                'character_distribution_anomalies': char_anomalies
            }
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            raise
            
    def analyze_vocabulary_trends(self, 
                                current_dataset: Dataset,
                                previous_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """
        Analyze vocabulary trends between datasets.
        
        Args:
            current_dataset: Current dataset
            previous_dataset: Previous dataset for comparison
            
        Returns:
            Dictionary containing vocabulary trend analysis
        """
        current_vocab = set(word for text in current_dataset[self.text_column] 
                          for word in text.lower().split())
        
        if previous_dataset is None:
            return {'vocabulary_size': len(current_vocab)}
            
        previous_vocab = set(word for text in previous_dataset[self.text_column] 
                           for word in text.lower().split())
        
        new_words = current_vocab - previous_vocab
        removed_words = previous_vocab - current_vocab
        
        return {
            'current_vocabulary_size': len(current_vocab),
            'previous_vocabulary_size': len(previous_vocab),
            'new_words': list(new_words),
            'removed_words': list(removed_words),
            'vocabulary_change_rate': (len(new_words) + len(removed_words)) / 
                                    len(previous_vocab)
        }
