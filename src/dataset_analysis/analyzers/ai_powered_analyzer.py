from typing import Dict, Any, List, Optional
from datasets import Dataset
import anthropic
import openai
import numpy as np
from datetime import datetime
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
from ..config.api_config import ANTHROPIC_API_KEY, OPENAI_API_KEY
from .base_analyzer import BaseAnalyzer
from ..learning.continuous_learner import ContinuousLearner

class AIPoweredAnalyzer(BaseAnalyzer):
    """Analyzer that uses both Anthropic and OpenAI models for enhanced analysis."""
    
    def __init__(self, text_column: str = 'text', storage_dir: str = './data/learning'):
        super().__init__()
        self.text_column = text_column
        self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.logger = logging.getLogger(__name__)
        self.continuous_learner = ContinuousLearner(storage_dir)
        
    async def analyze_with_claude(self, text: str) -> Dict[str, Any]:
        """Analyze text using Anthropic's Claude."""
        try:
            prompt = f"""Analyze the following text and provide:
1. Main themes and concepts
2. Key insights
3. Technical complexity level
4. Research implications
5. Potential applications

Text: {text}

Provide the analysis in JSON format with these keys:
themes, insights, complexity_level, implications, applications"""

            message = await self.anthropic_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                temperature=0,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Parse JSON response
            analysis = json.loads(message.content[0].text)
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in Claude analysis: {str(e)}")
            return {}
            
    async def analyze_with_gpt4(self, text: str) -> Dict[str, Any]:
        """Analyze text using OpenAI's GPT-4."""
        try:
            prompt = f"""Analyze the following text and provide:
1. Innovation assessment
2. Research gaps
3. Future directions
4. Potential challenges
5. Implementation considerations

Text: {text}

Provide the analysis in JSON format with these keys:
innovation_score, research_gaps, future_directions, challenges, implementation"""

            response = await self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )
            
            # Parse JSON response
            analysis = json.loads(response.choices[0].message.content)
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in GPT-4 analysis: {str(e)}")
            return {}
            
    async def analyze_async(self, dataset) -> Dict[str, Any]:
        """
        Asynchronous analysis with continuous learning.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary containing analysis results and learning outcomes
        """
        try:
            # Run initial analysis
            analysis_results = await self._run_initial_analysis(dataset)
            
            # Get dataset info
            dataset_info = self._get_dataset_info(dataset)
            
            # Apply continuous learning
            learning_outcomes = await self.continuous_learner.learn_from_analysis(
                analysis_results,
                dataset_info
            )
            
            # Enhance analysis with learned insights
            enhanced_results = await self._enhance_analysis(
                analysis_results,
                learning_outcomes
            )
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Error in async analysis: {str(e)}")
            raise
            
    def analyze(self, dataset) -> Dict[str, Any]:
        """
        Synchronous wrapper for analyze_async.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        return asyncio.run(self.analyze_async(dataset))
        
    async def _run_initial_analysis(self, dataset) -> Dict[str, Any]:
        """Run initial analysis using both AI models."""
        try:
            # Run analyses in parallel
            claude_task = self.analyze_with_claude(dataset[self.text_column][0])
            gpt4_task = self.analyze_with_gpt4(dataset[self.text_column][0])
            
            # Wait for both analyses
            claude_results, gpt4_results = await asyncio.gather(
                claude_task,
                gpt4_task
            )
            
            # Combine results
            combined_results = {
                'timestamp': datetime.now().isoformat(),
                'claude_analysis': claude_results,
                'gpt4_analysis': gpt4_results,
                'consensus_metrics': self._calculate_consensus_metrics(
                    claude_results,
                    gpt4_results
                )
            }
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Error in initial analysis: {str(e)}")
            raise
            
    def _get_dataset_info(self, dataset) -> Dict[str, Any]:
        """Get information about the dataset."""
        return {
            'size': len(dataset),
            'columns': list(dataset.features.keys()),
            'text_column': self.text_column,
            'timestamp': datetime.now().isoformat()
        }
        
    async def _enhance_analysis(self,
                              analysis_results: Dict[str, Any],
                              learning_outcomes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance analysis results with learned insights.
        
        Args:
            analysis_results: Initial analysis results
            learning_outcomes: Outcomes from continuous learning
            
        Returns:
            Enhanced analysis results
        """
        try:
            # Add learning outcomes
            enhanced_results = {
                **analysis_results,
                'learning_outcomes': learning_outcomes,
                'enhanced_metrics': {}
            }
            
            # Calculate enhanced metrics using learned patterns
            enhanced_results['enhanced_metrics'] = await self._calculate_enhanced_metrics(
                analysis_results,
                learning_outcomes
            )
            
            # Add improvement suggestions
            enhanced_results['suggestions'] = learning_outcomes.get(
                'suggestions',
                {}
            )
            
            # Add confidence scores
            enhanced_results['confidence_scores'] = self._calculate_confidence_scores(
                analysis_results,
                learning_outcomes
            )
            
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"Error enhancing analysis: {str(e)}")
            return analysis_results
            
    async def _calculate_enhanced_metrics(self,
                                       analysis_results: Dict[str, Any],
                                       learning_outcomes: Dict[str, Any]
                                       ) -> Dict[str, Any]:
        """Calculate enhanced metrics using learned patterns."""
        try:
            # Get learned patterns
            patterns = learning_outcomes.get('patterns', {})
            
            # Calculate pattern-based metrics
            pattern_metrics = {
                'pattern_match_score': self._calculate_pattern_match_score(
                    analysis_results,
                    patterns
                ),
                'insight_relevance_score': self._calculate_insight_relevance_score(
                    analysis_results,
                    learning_outcomes.get('insights', {})
                ),
                'adaptation_score': self._calculate_adaptation_score(
                    analysis_results,
                    learning_outcomes
                )
            }
            
            return pattern_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced metrics: {str(e)}")
            return {}
            
    def _calculate_confidence_scores(self,
                                   analysis_results: Dict[str, Any],
                                   learning_outcomes: Dict[str, Any]
                                   ) -> Dict[str, Any]:
        """Calculate confidence scores based on analysis and learning outcomes."""
        try:
            return {
                'analysis_confidence': self._calculate_analysis_confidence(
                    analysis_results
                ),
                'learning_confidence': self._calculate_learning_confidence(
                    learning_outcomes
                ),
                'combined_confidence': self._calculate_combined_confidence(
                    analysis_results,
                    learning_outcomes
                )
            }
        except Exception as e:
            self.logger.error(f"Error calculating confidence scores: {str(e)}")
            return {}
            
    def _calculate_pattern_match_score(self,
                                     analysis_results: Dict[str, Any],
                                     patterns: Dict[str, Any]) -> float:
        """Calculate how well current results match learned patterns."""
        try:
            if not patterns:
                return 0.0
                
            # Extract pattern features
            pattern_features = patterns.get('features', [])
            
            # Calculate match score
            matches = sum(
                1 for feature in pattern_features
                if feature in str(analysis_results)
            )
            
            return matches / len(pattern_features) if pattern_features else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern match score: {str(e)}")
            return 0.0
            
    def _calculate_insight_relevance_score(self,
                                         analysis_results: Dict[str, Any],
                                         insights: Dict[str, Any]) -> float:
        """Calculate relevance of learned insights to current analysis."""
        try:
            if not insights:
                return 0.0
                
            # Extract insight keywords
            keywords = insights.get('keywords', [])
            
            # Calculate relevance score
            matches = sum(
                1 for keyword in keywords
                if keyword.lower() in str(analysis_results).lower()
            )
            
            return matches / len(keywords) if keywords else 0.0
            
        except Exception as e:
            self.logger.error(
                f"Error calculating insight relevance score: {str(e)}"
            )
            return 0.0
            
    def _calculate_adaptation_score(self,
                                  analysis_results: Dict[str, Any],
                                  learning_outcomes: Dict[str, Any]) -> float:
        """Calculate how well the analysis has adapted based on learning."""
        try:
            if not learning_outcomes:
                return 0.0
                
            # Get adaptation metrics
            adaptations = learning_outcomes.get('suggestions', {}).get(
                'adaptations',
                []
            )
            
            if not adaptations:
                return 0.0
                
            # Calculate adaptation score
            implemented = sum(
                1 for adaptation in adaptations
                if adaptation['status'] == 'implemented'
            )
            
            return implemented / len(adaptations)
            
        except Exception as e:
            self.logger.error(f"Error calculating adaptation score: {str(e)}")
            return 0.0
            
    def _calculate_analysis_confidence(self,
                                     analysis_results: Dict[str, Any]) -> float:
        """Calculate confidence in analysis results."""
        try:
            confidence_factors = [
                analysis_results.get('consensus_metrics', {}).get(
                    'agreement_score',
                    0
                ),
                analysis_results.get('claude_analysis', {}).get(
                    'confidence',
                    0
                ),
                analysis_results.get('gpt4_analysis', {}).get(
                    'confidence',
                    0
                )
            ]
            
            return np.mean([f for f in confidence_factors if f is not None])
            
        except Exception as e:
            self.logger.error(f"Error calculating analysis confidence: {str(e)}")
            return 0.0
            
    def _calculate_learning_confidence(self,
                                     learning_outcomes: Dict[str, Any]) -> float:
        """Calculate confidence in learning outcomes."""
        try:
            confidence_factors = [
                learning_outcomes.get('insights', {}).get('confidence', 0),
                learning_outcomes.get('patterns', {}).get('reliability', 0),
                learning_outcomes.get('suggestions', {}).get('confidence', 0)
            ]
            
            return np.mean([f for f in confidence_factors if f is not None])
            
        except Exception as e:
            self.logger.error(f"Error calculating learning confidence: {str(e)}")
            return 0.0
            
    def _calculate_combined_confidence(self,
                                     analysis_results: Dict[str, Any],
                                     learning_outcomes: Dict[str, Any]) -> float:
        """Calculate combined confidence score."""
        try:
            analysis_conf = self._calculate_analysis_confidence(analysis_results)
            learning_conf = self._calculate_learning_confidence(learning_outcomes)
            
            # Weight learning confidence slightly higher (0.6 vs 0.4)
            return (0.4 * analysis_conf) + (0.6 * learning_conf)
            
        except Exception as e:
            self.logger.error(f"Error calculating combined confidence: {str(e)}")
            return 0.0

    def detect_anomalies(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Detect anomalies using AI models.
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary containing detected anomalies
        """
        try:
            # Sample texts for analysis
            sample_size = min(5, len(dataset))
            sample_indices = np.random.choice(
                len(dataset), sample_size, replace=False
            )
            sample_texts = [dataset[self.text_column][i] for i in sample_indices]
            
            anomalies = {
                'content_anomalies': [],
                'complexity_anomalies': [],
                'consistency_anomalies': []
            }
            
            async def analyze_anomalies(text: str):
                # Ask Claude to identify potential anomalies
                prompt = f"""Analyze this text for potential anomalies in:
1. Content consistency
2. Technical accuracy
3. Logical flow
4. Terminology usage

Text: {text}

Provide analysis in JSON format with these keys:
content_issues, technical_issues, logical_issues, terminology_issues"""

                message = await self.anthropic_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=500,
                    temperature=0,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }]
                )
                
                return json.loads(message.content[0].text)
            
            async def analyze_all_anomalies():
                tasks = [analyze_anomalies(text) for text in sample_texts]
                return await asyncio.gather(*tasks)
            
            # Run anomaly detection
            anomaly_results = asyncio.run(analyze_all_anomalies())
            
            # Process results
            for result in anomaly_results:
                for issue_type, issues in result.items():
                    if issues:  # If any issues were found
                        anomalies[issue_type].extend(issues)
            
            return anomalies
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            raise
            
    @staticmethod
    def _aggregate_list_items(items: List[List[str]]) -> List[str]:
        """Aggregate and deduplicate list items."""
        flat_list = [item for sublist in items for item in sublist]
        return list(set(flat_list))
        
    @staticmethod
    def _calculate_average(values: List[float]) -> float:
        """Calculate average of numeric values, ignoring None."""
        valid_values = [v for v in values if v is not None]
        return np.mean(valid_values) if valid_values else 0
        
    def _calculate_innovation_potential(self,
                                     claude_analysis: Dict[str, Any],
                                     gpt4_analysis: Dict[str, Any]) -> float:
        """Calculate innovation potential score."""
        factors = [
            len(claude_analysis['insights']) / 10,
            len(claude_analysis['applications']) / 5,
            gpt4_analysis['innovation_scores'] / 10,
            (10 - len(gpt4_analysis['research_gaps'])) / 10
        ]
        return np.mean(factors) * 10
        
    def _calculate_research_maturity(self,
                                   claude_analysis: Dict[str, Any],
                                   gpt4_analysis: Dict[str, Any]) -> float:
        """Calculate research maturity score."""
        factors = [
            claude_analysis['complexity_levels'] / 10,
            len(claude_analysis['implications']) / 5,
            (10 - len(gpt4_analysis['research_gaps'])) / 10,
            len(gpt4_analysis['implementation']) / 5
        ]
        return np.mean(factors) * 10
        
    def _calculate_implementation_feasibility(self,
                                           claude_analysis: Dict[str, Any],
                                           gpt4_analysis: Dict[str, Any]) -> float:
        """Calculate implementation feasibility score."""
        factors = [
            len(claude_analysis['applications']) / 5,
            (10 - claude_analysis['complexity_levels']) / 10,
            (10 - len(gpt4_analysis['challenges'])) / 10,
            len(gpt4_analysis['implementation']) / 5
        ]
        return np.mean(factors) * 10
