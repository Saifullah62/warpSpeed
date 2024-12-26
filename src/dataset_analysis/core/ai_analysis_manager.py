from typing import Dict, Any, List, Optional
import asyncio
from datetime import datetime
import logging
from ..analyzers.ai_powered_analyzer import AIPoweredAnalyzer
from ..analyzers.text_analyzer import TextAnalyzer

class AIAnalysisManager:
    """Manages and coordinates AI-powered analysis with traditional analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ai_analyzer = AIPoweredAnalyzer()
        self.text_analyzer = TextAnalyzer()
        
    async def run_comprehensive_analysis(self, 
                                      dataset,
                                      text_column: str = 'text') -> Dict[str, Any]:
        """
        Run both AI-powered and traditional analysis in parallel.
        
        Args:
            dataset: Dataset to analyze
            text_column: Name of the text column
            
        Returns:
            Dictionary containing combined analysis results
        """
        try:
            # Run analyses in parallel
            ai_analysis_task = asyncio.create_task(
                self._run_ai_analysis(dataset, text_column)
            )
            traditional_analysis_task = asyncio.create_task(
                self._run_traditional_analysis(dataset, text_column)
            )
            
            # Wait for both analyses to complete
            ai_results, traditional_results = await asyncio.gather(
                ai_analysis_task,
                traditional_analysis_task
            )
            
            # Combine results
            combined_results = self._combine_analyses(
                ai_results,
                traditional_results
            )
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {str(e)}")
            raise
            
    async def _run_ai_analysis(self,
                              dataset,
                              text_column: str) -> Dict[str, Any]:
        """Run AI-powered analysis."""
        try:
            return await self.ai_analyzer.analyze_async(dataset)
        except Exception as e:
            self.logger.error(f"Error in AI analysis: {str(e)}")
            return {}
            
    async def _run_traditional_analysis(self,
                                      dataset,
                                      text_column: str) -> Dict[str, Any]:
        """Run traditional text analysis."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self.text_analyzer.analyze, dataset
            )
        except Exception as e:
            self.logger.error(f"Error in traditional analysis: {str(e)}")
            return {}
            
    def _combine_analyses(self,
                         ai_results: Dict[str, Any],
                         traditional_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine AI and traditional analysis results.
        
        Args:
            ai_results: Results from AI analysis
            traditional_results: Results from traditional analysis
            
        Returns:
            Combined analysis results
        """
        combined = {
            'timestamp': datetime.now(),
            'ai_analysis': ai_results,
            'traditional_analysis': traditional_results,
            'enhanced_metrics': {}
        }
        
        # Calculate enhanced metrics using both analyses
        combined['enhanced_metrics'] = self._calculate_enhanced_metrics(
            ai_results,
            traditional_results
        )
        
        return combined
        
    def _calculate_enhanced_metrics(self,
                                  ai_results: Dict[str, Any],
                                  traditional_results: Dict[str, Any]
                                  ) -> Dict[str, Any]:
        """Calculate enhanced metrics using both analyses."""
        enhanced_metrics = {}
        
        try:
            # Combine sentiment analysis
            if 'sentiment_distribution' in traditional_results:
                trad_sentiment = traditional_results['sentiment_distribution']
                ai_sentiment = ai_results.get('claude_analysis', {}).get(
                    'sentiment', {}
                )
                
                enhanced_metrics['sentiment_confidence'] = self._calculate_sentiment_confidence(
                    trad_sentiment,
                    ai_sentiment
                )
            
            # Enhance topic analysis
            if 'topics' in traditional_results:
                trad_topics = traditional_results['topics']
                ai_themes = ai_results.get('claude_analysis', {}).get(
                    'themes', []
                )
                
                enhanced_metrics['topic_coverage'] = self._calculate_topic_coverage(
                    trad_topics,
                    ai_themes
                )
            
            # Calculate overall confidence scores
            enhanced_metrics['analysis_confidence'] = {
                'ai_confidence': self._calculate_ai_confidence(ai_results),
                'traditional_confidence': self._calculate_traditional_confidence(
                    traditional_results
                ),
                'combined_confidence': self._calculate_combined_confidence(
                    ai_results,
                    traditional_results
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced metrics: {str(e)}")
            
        return enhanced_metrics
        
    def _calculate_sentiment_confidence(self,
                                     trad_sentiment: Dict[str, float],
                                     ai_sentiment: Dict[str, float]) -> float:
        """Calculate confidence in sentiment analysis."""
        if not trad_sentiment or not ai_sentiment:
            return 0.0
            
        # Calculate agreement between traditional and AI sentiment
        total_diff = sum(
            abs(trad_sentiment.get(k, 0) - ai_sentiment.get(k, 0))
            for k in set(trad_sentiment) | set(ai_sentiment)
        )
        
        return 1.0 - (total_diff / max(
            sum(trad_sentiment.values()),
            sum(ai_sentiment.values())
        ))
        
    def _calculate_topic_coverage(self,
                                trad_topics: List[str],
                                ai_themes: List[str]) -> Dict[str, Any]:
        """Calculate topic coverage metrics."""
        trad_set = set(trad_topics)
        ai_set = set(ai_themes)
        
        return {
            'overlap': len(trad_set & ai_set),
            'unique_traditional': len(trad_set - ai_set),
            'unique_ai': len(ai_set - trad_set),
            'coverage_ratio': len(trad_set & ai_set) / len(trad_set | ai_set)
        }
        
    def _calculate_ai_confidence(self,
                               ai_results: Dict[str, Any]) -> float:
        """Calculate confidence in AI analysis."""
        if not ai_results:
            return 0.0
            
        confidence_factors = [
            len(ai_results.get('claude_analysis', {}).get('insights', [])) / 10,
            ai_results.get('consensus_metrics', {}).get(
                'research_maturity', 0
            ) / 10,
            ai_results.get('consensus_metrics', {}).get(
                'implementation_feasibility', 0
            ) / 10
        ]
        
        return sum(confidence_factors) / len(confidence_factors)
        
    def _calculate_traditional_confidence(self,
                                       trad_results: Dict[str, Any]) -> float:
        """Calculate confidence in traditional analysis."""
        if not trad_results:
            return 0.0
            
        confidence_factors = [
            min(len(trad_results.get('topics', [])) / 5, 1.0),
            min(trad_results.get('vocabulary_size', 0) / 1000, 1.0),
            min(len(trad_results.get('sentiment_distribution', {})) / 3, 1.0)
        ]
        
        return sum(confidence_factors) / len(confidence_factors)
        
    def _calculate_combined_confidence(self,
                                     ai_results: Dict[str, Any],
                                     trad_results: Dict[str, Any]) -> float:
        """Calculate combined analysis confidence."""
        ai_conf = self._calculate_ai_confidence(ai_results)
        trad_conf = self._calculate_traditional_confidence(trad_results)
        
        # Weight AI confidence slightly higher (0.6 vs 0.4)
        return (0.6 * ai_conf) + (0.4 * trad_conf)
