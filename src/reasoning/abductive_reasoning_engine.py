import numpy as np
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import itertools
import scipy.stats as stats

@dataclass
class Hypothesis:
    """
    Represents a generated hypothesis with probabilistic properties
    """
    id: str
    description: str
    probability: float = 0.0
    explanatory_power: float = 0.0
    confidence_interval: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    supporting_evidence: List[Dict[str, Any]] = field(default_factory=list)
    contradicting_evidence: List[Dict[str, Any]] = field(default_factory=list)

class AbductiveReasoningEngine:
    def __init__(self, causal_reasoning_engine=None, knowledge_graph=None):
        """
        Initialize Abductive Reasoning Engine
        
        Args:
            causal_reasoning_engine: Optional causal reasoning engine
            knowledge_graph: Optional knowledge graph
        """
        # Hypothesis management
        self.hypothesis_graph = nx.DiGraph()
        self.generated_hypotheses = {}
        
        # Reasoning dependencies
        self.causal_reasoning_engine = causal_reasoning_engine
        self.knowledge_graph = knowledge_graph
        
        # Reasoning parameters
        self.max_hypothesis_generation_depth = 3
        self.hypothesis_probability_threshold = 0.6
        self.explanatory_power_weight = 0.7
    
    def generate_hypotheses(
        self, 
        observations: List[Dict[str, Any]], 
        context: Dict[str, Any] = None
    ) -> List[Hypothesis]:
        """
        Generate hypotheses to explain given observations
        
        Args:
            observations: List of observed phenomena
            context: Additional contextual information
        
        Returns:
            List of generated hypotheses
        """
        # Preprocess observations
        processed_observations = self._preprocess_observations(observations)
        
        # Generate initial hypotheses
        hypotheses = []
        for observation in processed_observations:
            # Generate hypotheses for each observation
            obs_hypotheses = self._generate_observation_hypotheses(
                observation, 
                context or {}
            )
            hypotheses.extend(obs_hypotheses)
        
        # Rank and filter hypotheses
        ranked_hypotheses = self._rank_hypotheses(hypotheses)
        
        # Store and track generated hypotheses
        for hypothesis in ranked_hypotheses:
            self.hypothesis_graph.add_node(hypothesis.id, hypothesis=hypothesis)
            self.generated_hypotheses[hypothesis.id] = hypothesis
        
        return ranked_hypotheses
    
    def _preprocess_observations(
        self, 
        observations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Preprocess and normalize observations
        
        Args:
            observations: Raw observations
        
        Returns:
            Processed observations
        """
        processed_observations = []
        for obs in observations:
            # Normalize and enrich observation
            processed_obs = {
                'id': obs.get('id', str(hash(str(obs)))),
                'description': obs.get('description', ''),
                'domain': obs.get('domain', 'unknown'),
                'timestamp': obs.get('timestamp'),
                'metadata': obs.get('metadata', {})
            }
            processed_observations.append(processed_obs)
        
        return processed_observations
    
    def _generate_observation_hypotheses(
        self, 
        observation: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> List[Hypothesis]:
        """
        Generate hypotheses for a specific observation
        
        Args:
            observation: Processed observation
            context: Contextual information
        
        Returns:
            List of generated hypotheses
        """
        hypotheses = []
        
        # Leverage causal reasoning if available
        if self.causal_reasoning_engine:
            # Find potential causal paths
            causal_paths = self._explore_causal_paths(observation)
            
            for path in causal_paths:
                hypothesis = self._construct_hypothesis_from_path(
                    observation, 
                    path, 
                    context
                )
                hypotheses.append(hypothesis)
        
        # Knowledge graph-based hypothesis generation
        if self.knowledge_graph:
            kg_hypotheses = self._generate_knowledge_graph_hypotheses(
                observation, 
                context
            )
            hypotheses.extend(kg_hypotheses)
        
        # Fallback: generate generic hypotheses
        generic_hypotheses = self._generate_generic_hypotheses(observation)
        hypotheses.extend(generic_hypotheses)
        
        return hypotheses
    
    def _explore_causal_paths(
        self, 
        observation: Dict[str, Any]
    ) -> List[List[Dict[str, Any]]]:
        """
        Explore potential causal paths related to the observation
        
        Args:
            observation: Processed observation
        
        Returns:
            List of potential causal paths
        """
        # Use causal reasoning to find potential paths
        if not self.causal_reasoning_engine:
            return []
        
        # Find related nodes in causal graph
        related_nodes = [
            node for node in self.causal_reasoning_engine.causal_graph.nodes
            if observation['domain'] in str(node).lower()
        ]
        
        causal_paths = []
        for source in related_nodes:
            for target in related_nodes:
                if source != target:
                    paths = self.causal_reasoning_engine.analyze_causal_paths(
                        source, 
                        target, 
                        max_path_length=self.max_hypothesis_generation_depth
                    )
                    causal_paths.extend(paths)
        
        return causal_paths
    
    def _construct_hypothesis_from_path(
        self, 
        observation: Dict[str, Any], 
        causal_path: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> Hypothesis:
        """
        Construct a hypothesis from a causal path
        
        Args:
            observation: Original observation
            causal_path: Causal path information
            context: Contextual information
        
        Returns:
            Generated hypothesis
        """
        # Compute hypothesis probability
        path_strength = causal_path.get('total_strength', 0.5)
        
        # Generate unique hypothesis ID
        hypothesis_id = f"hyp_{hash(str(causal_path))}"
        
        # Construct hypothesis description
        path_nodes = causal_path.get('path', [])
        description = f"Explains {observation['description']} through path: {' -> '.join(path_nodes)}"
        
        # Compute probability and explanatory power
        probability = path_strength * (1 + len(path_nodes) / 10)
        explanatory_power = self._compute_explanatory_power(
            observation, 
            causal_path, 
            context
        )
        
        # Compute confidence interval
        confidence_interval = self._compute_confidence_interval(probability)
        
        return Hypothesis(
            id=hypothesis_id,
            description=description,
            probability=probability,
            explanatory_power=explanatory_power,
            confidence_interval=confidence_interval
        )
    
    def _generate_knowledge_graph_hypotheses(
        self, 
        observation: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> List[Hypothesis]:
        """
        Generate hypotheses using knowledge graph
        
        Args:
            observation: Processed observation
            context: Contextual information
        
        Returns:
            List of generated hypotheses
        """
        if not self.knowledge_graph:
            return []
        
        hypotheses = []
        
        # Find related entities in knowledge graph
        related_entities = [
            entity for entity in self.knowledge_graph.nodes
            if observation['domain'] in str(entity).lower()
        ]
        
        for entity in related_entities:
            # Generate hypothesis based on entity properties
            hypothesis_id = f"kg_hyp_{hash(str(entity))}"
            description = f"Explains {observation['description']} through knowledge graph entity: {entity}"
            
            # Compute basic probability
            probability = np.random.uniform(0.3, 0.7)
            explanatory_power = self._compute_explanatory_power(
                observation, 
                {'entity': entity}, 
                context
            )
            
            confidence_interval = self._compute_confidence_interval(probability)
            
            hypothesis = Hypothesis(
                id=hypothesis_id,
                description=description,
                probability=probability,
                explanatory_power=explanatory_power,
                confidence_interval=confidence_interval
            )
            
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_generic_hypotheses(
        self, 
        observation: Dict[str, Any]
    ) -> List[Hypothesis]:
        """
        Generate generic fallback hypotheses
        
        Args:
            observation: Processed observation
        
        Returns:
            List of generic hypotheses
        """
        generic_hypotheses = []
        
        # Generate multiple generic hypotheses
        hypothesis_templates = [
            f"Random occurrence in {observation['domain']} domain",
            f"Emergent phenomenon related to {observation['description']}",
            "Unexplained complex system interaction"
        ]
        
        for template in hypothesis_templates:
            hypothesis_id = f"generic_hyp_{hash(template)}"
            
            hypothesis = Hypothesis(
                id=hypothesis_id,
                description=template,
                probability=np.random.uniform(0.1, 0.4),
                explanatory_power=np.random.uniform(0.2, 0.5),
                confidence_interval=(0.1, 0.5)
            )
            
            generic_hypotheses.append(hypothesis)
        
        return generic_hypotheses
    
    def _compute_explanatory_power(
        self, 
        observation: Dict[str, Any], 
        hypothesis_context: Dict[str, Any], 
        global_context: Dict[str, Any]
    ) -> float:
        """
        Compute explanatory power of a hypothesis
        
        Args:
            observation: Original observation
            hypothesis_context: Context specific to the hypothesis
            global_context: Overall contextual information
        
        Returns:
            Explanatory power score
        """
        # Compute contextual similarity
        context_similarity = self._compute_context_similarity(
            observation, 
            hypothesis_context, 
            global_context
        )
        
        # Compute domain relevance
        domain_relevance = self._compute_domain_relevance(
            observation, 
            hypothesis_context
        )
        
        # Weighted combination of factors
        explanatory_power = (
            self.explanatory_power_weight * context_similarity + 
            (1 - self.explanatory_power_weight) * domain_relevance
        )
        
        return explanatory_power
    
    def _compute_context_similarity(
        self, 
        observation: Dict[str, Any], 
        hypothesis_context: Dict[str, Any], 
        global_context: Dict[str, Any]
    ) -> float:
        """
        Compute context similarity between observation and hypothesis
        
        Args:
            observation: Original observation
            hypothesis_context: Hypothesis-specific context
            global_context: Overall contextual information
        
        Returns:
            Context similarity score
        """
        # Simple context similarity computation
        similarity_factors = [
            observation.get('domain', ''),
            observation.get('description', ''),
            str(hypothesis_context),
            str(global_context)
        ]
        
        # Use hash-based similarity
        similarity_hash = hash(''.join(similarity_factors)) % 1000 / 1000.0
        
        return similarity_hash
    
    def _compute_domain_relevance(
        self, 
        observation: Dict[str, Any], 
        hypothesis_context: Dict[str, Any]
    ) -> float:
        """
        Compute domain relevance between observation and hypothesis
        
        Args:
            observation: Original observation
            hypothesis_context: Hypothesis-specific context
        
        Returns:
            Domain relevance score
        """
        # Check domain alignment
        obs_domain = observation.get('domain', 'unknown')
        hyp_context_str = str(hypothesis_context).lower()
        
        domain_match = obs_domain.lower() in hyp_context_str
        
        return 1.0 if domain_match else np.random.uniform(0.3, 0.7)
    
    def _compute_confidence_interval(
        self, 
        probability: float
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for a probability
        
        Args:
            probability: Base probability
        
        Returns:
            Confidence interval tuple
        """
        # Compute standard error
        standard_error = np.sqrt(probability * (1 - probability))
        
        # 95% confidence interval
        lower_bound = max(0, probability - 1.96 * standard_error)
        upper_bound = min(1, probability + 1.96 * standard_error)
        
        return (lower_bound, upper_bound)
    
    def _rank_hypotheses(
        self, 
        hypotheses: List[Hypothesis]
    ) -> List[Hypothesis]:
        """
        Rank and filter hypotheses
        
        Args:
            hypotheses: List of generated hypotheses
        
        Returns:
            Ranked and filtered hypotheses
        """
        # Compute composite score
        scored_hypotheses = []
        for hypothesis in hypotheses:
            composite_score = (
                self.explanatory_power_weight * hypothesis.explanatory_power +
                (1 - self.explanatory_power_weight) * hypothesis.probability
            )
            
            scored_hypotheses.append((composite_score, hypothesis))
        
        # Sort hypotheses by composite score
        ranked_hypotheses = [
            hyp for _, hyp in sorted(
                scored_hypotheses, 
                key=lambda x: x[0], 
                reverse=True
            )
            if hyp.probability >= self.hypothesis_probability_threshold
        ]
        
        return ranked_hypotheses[:10]  # Limit to top 10 hypotheses
    
    def evaluate_hypothesis(
        self, 
        hypothesis_id: str, 
        new_evidence: Dict[str, Any]
    ) -> Hypothesis:
        """
        Evaluate and update a hypothesis with new evidence
        
        Args:
            hypothesis_id: ID of the hypothesis to evaluate
            new_evidence: New evidence to consider
        
        Returns:
            Updated hypothesis
        """
        if hypothesis_id not in self.generated_hypotheses:
            raise ValueError(f"Hypothesis {hypothesis_id} not found")
        
        hypothesis = self.generated_hypotheses[hypothesis_id]
        
        # Update supporting and contradicting evidence
        evidence_relevance = self._compute_evidence_relevance(
            hypothesis, 
            new_evidence
        )
        
        if evidence_relevance > 0.5:
            hypothesis.supporting_evidence.append(new_evidence)
            # Recompute probability based on supporting evidence
            hypothesis.probability *= 1.2
        else:
            hypothesis.contradicting_evidence.append(new_evidence)
            # Reduce probability for contradicting evidence
            hypothesis.probability *= 0.8
        
        # Recompute confidence interval
        hypothesis.confidence_interval = self._compute_confidence_interval(
            hypothesis.probability
        )
        
        # Update in hypothesis graph and dictionary
        self.generated_hypotheses[hypothesis_id] = hypothesis
        self.hypothesis_graph.nodes[hypothesis_id]['hypothesis'] = hypothesis
        
        return hypothesis
    
    def _compute_evidence_relevance(
        self, 
        hypothesis: Hypothesis, 
        evidence: Dict[str, Any]
    ) -> float:
        """
        Compute relevance of new evidence to a hypothesis
        
        Args:
            hypothesis: Target hypothesis
            evidence: New evidence
        
        Returns:
            Evidence relevance score
        """
        # Compare evidence description with hypothesis description
        description_similarity = len(
            set(hypothesis.description.lower().split()) & 
            set(str(evidence).lower().split())
        ) / len(hypothesis.description.split())
        
        return description_similarity
