"""Performance benchmarks for critical system components."""

import pytest
import asyncio
import numpy as np
from typing import Dict, Any

from src.knowledge_graph.distributed_quantum_graph import DistributedQuantumGraph
from src.semantic_understanding.multilingual_semantic_engine import MultilingualSemanticEngine
from src.interaction.psychological_profile_engine import PsychologicalProfileEngine
from src.reasoning.explainable_reasoning_engine import ExplainableReasoningEngine
from src.integration.performance_monitoring import PerformanceMonitor

@pytest.fixture
def performance_monitor():
    """Initialize performance monitor for benchmarks."""
    return PerformanceMonitor()

@pytest.fixture
def quantum_graph():
    """Initialize quantum graph for benchmarks."""
    return DistributedQuantumGraph()

@pytest.fixture
def semantic_engine():
    """Initialize semantic engine for benchmarks."""
    return MultilingualSemanticEngine()

@pytest.fixture
def profile_engine():
    """Initialize profile engine for benchmarks."""
    return PsychologicalProfileEngine()

@pytest.fixture
def reasoning_engine():
    """Initialize reasoning engine for benchmarks."""
    return ExplainableReasoningEngine()

def test_quantum_graph_operations(benchmark, quantum_graph, performance_monitor):
    """Benchmark quantum graph operations."""
    def add_nodes():
        with performance_monitor.component_timer('quantum_graph'):
            for i in range(100):
                quantum_graph.add_node(
                    node_id=f'test_node_{i}',
                    state_vector=np.random.rand(8),
                    properties={'test': True}
                )
    
    result = benchmark(add_nodes)
    metrics = performance_monitor.get_component_metrics('quantum_graph')
    assert metrics['latency']['avg'] < 100  # Less than 100ms per operation

def test_semantic_processing(benchmark, semantic_engine, performance_monitor):
    """Benchmark semantic processing operations."""
    test_texts = [
        "Quantum computing leverages quantum mechanics for computation",
        "Machine learning algorithms learn patterns from data",
        "Neural networks process information through layers"
    ]
    
    def process_texts():
        with performance_monitor.component_timer('semantic_engine'):
            for text in test_texts:
                semantic_engine.analyze_text(
                    text=text,
                    source_lang='en'
                )
    
    result = benchmark(process_texts)
    metrics = performance_monitor.get_component_metrics('semantic_engine')
    assert metrics['latency']['p95'] < 200  # 95th percentile under 200ms

def test_profile_engine_performance(benchmark, profile_engine, performance_monitor):
    """Benchmark psychological profile engine operations."""
    test_assessment = {
        'analytical': 0.8,
        'intuitive': 0.6,
        'visual': 0.7,
        'verbal': 0.5
    }
    
    def create_profiles():
        with performance_monitor.component_timer('profile_engine'):
            for i in range(50):
                profile_engine.create_user_profile(
                    user_id=f'test_user_{i}',
                    initial_assessment=test_assessment
                )
    
    result = benchmark(create_profiles)
    metrics = performance_monitor.get_component_metrics('profile_engine')
    assert metrics['latency']['avg'] < 50  # Less than 50ms per profile

@pytest.mark.asyncio
async def test_reasoning_engine_performance(benchmark, reasoning_engine, performance_monitor):
    """Benchmark reasoning engine operations."""
    test_hypothesis = "Quantum entanglement enables faster-than-light communication"
    test_evidence = [
        "Quantum entanglement has been experimentally verified",
        "Information transfer appears instantaneous",
        "No classical signal is transmitted"
    ]
    
    def generate_explanation():
        with performance_monitor.component_timer('reasoning_engine'):
            reasoning_engine.explain_hypothesis(
                hypothesis=test_hypothesis,
                evidence=test_evidence
            )
    
    result = benchmark(generate_explanation)
    metrics = performance_monitor.get_component_metrics('reasoning_engine')
    assert metrics['latency']['p99'] < 300  # 99th percentile under 300ms

def test_system_integration_performance(
    benchmark,
    quantum_graph,
    semantic_engine,
    profile_engine,
    reasoning_engine,
    performance_monitor
):
    """Benchmark full system integration performance."""
    def integrated_operation():
        with performance_monitor.component_timer('system_integration'):
            # Simulate complete processing pipeline
            text = "Quantum computing enables parallel computation through superposition"
            
            # 1. Semantic analysis
            semantic_repr = semantic_engine.analyze_text(text, source_lang='en')
            
            # 2. Knowledge graph update
            node_id = 'test_concept'
            quantum_graph.add_node(
                node_id=node_id,
                state_vector=semantic_repr.embedding.numpy(),
                properties={'type': 'concept'}
            )
            
            # 3. Profile update
            profile = profile_engine.create_user_profile(
                user_id='test_user',
                initial_assessment={'analytical': 0.8}
            )
            
            # 4. Reasoning
            reasoning_engine.explain_hypothesis(
                hypothesis=text,
                evidence=["Quantum superposition is a fundamental principle"]
            )
    
    result = benchmark(integrated_operation)
    metrics = performance_monitor.get_component_metrics('system_integration')
    assert metrics['latency']['avg'] < 500  # Complete pipeline under 500ms

def test_memory_usage(performance_monitor):
    """Test memory usage across operations."""
    initial_memory = performance_monitor.get_system_metrics('total_memory')[-1]
    
    # Perform memory-intensive operations
    large_data = [np.random.rand(1000, 1000) for _ in range(10)]
    
    final_memory = performance_monitor.get_system_metrics('total_memory')[-1]
    memory_increase = final_memory - initial_memory
    
    assert memory_increase < 1000  # Less than 1GB increase

def test_cpu_utilization(performance_monitor):
    """Test CPU utilization across operations."""
    cpu_usage = performance_monitor.get_system_metrics('cpu_usage')
    avg_cpu = np.mean(cpu_usage)
    
    assert avg_cpu < 80  # CPU usage should stay under 80%
