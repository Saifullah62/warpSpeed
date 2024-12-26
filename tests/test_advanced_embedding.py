import asyncio
import os
import pytest
import numpy as np
import torch

# Import advanced embedding components
from src.knowledge_graph.advanced_embedding import (
    MultiModalEmbeddingFinetuner,
    AdaptiveEmbeddingOptimizer,
    MetaLearningEmbeddingAdapter,
    MultiModalTransformerEmbedding,
    quantize_embedding_model,
    prune_embedding_model,
    PerformanceMetricsTracker,
    AdaptiveFinetuningStrategy
)
from src.knowledge_graph.schema import Entity, EntityType

@pytest.fixture
def multi_modal_embedding_finetuner():
    """
    Fixture for creating a multi-modal embedding fine-tuner.
    """
    return MultiModalEmbeddingFinetuner()

@pytest.fixture
def adaptive_embedding_optimizer():
    """
    Fixture for creating an adaptive embedding optimizer.
    """
    return AdaptiveEmbeddingOptimizer()

@pytest.fixture
def meta_learning_embedding_adapter():
    """
    Fixture for creating a meta-learning embedding adapter.
    """
    return MetaLearningEmbeddingAdapter()

@pytest.fixture
def multi_modal_transformer_embedding():
    """
    Fixture for creating a multi-modal transformer embedding.
    """
    return MultiModalTransformerEmbedding()

@pytest.fixture
def performance_tracker():
    """
    Create a performance metrics tracker for testing.
    """
    return PerformanceMetricsTracker(task_name="test_embedding_task")

class TestMultiModalEmbeddingFinetuner:
    @pytest.mark.asyncio
    async def test_text_embedding_preprocessing(self, multi_modal_embedding_finetuner):
        """
        Test text embedding preprocessing.
        
        Validates:
        - Successful text embedding generation
        - Embedding vector properties
        """
        # Sample scientific text
        text = "Quantum mechanics explores fundamental principles of nature"
        
        # Preprocess text
        text_embedding = multi_modal_embedding_finetuner._preprocess_text(text)
        
        # Validate embedding
        assert isinstance(text_embedding, torch.Tensor), "Embedding should be a torch tensor"
        assert text_embedding.dim() == 2, "Embedding should be a 2D tensor"
        assert text_embedding.size(1) == 768, "Unexpected embedding dimension"
    
    @pytest.mark.skipif(
        not os.path.exists('tests/test_images/quantum_diagram.jpg'), 
        reason="Test image not available"
    )
    def test_image_embedding_preprocessing(self, multi_modal_embedding_finetuner):
        """
        Test image embedding preprocessing.
        
        Validates:
        - Successful image embedding generation
        - Embedding vector properties
        """
        # Sample scientific image
        image_path = 'tests/test_images/quantum_diagram.jpg'
        
        # Preprocess image
        image_embedding = multi_modal_embedding_finetuner._preprocess_image(image_path)
        
        # Validate embedding
        assert isinstance(image_embedding, torch.Tensor), "Embedding should be a torch tensor"
        assert image_embedding.dim() == 1, "Embedding should be a 1D tensor"
        assert image_embedding.size(0) == 2048, "Unexpected embedding dimension"
    
    @pytest.mark.asyncio
    async def test_fine_tune_embeddings(self, multi_modal_embedding_finetuner):
        """
        Test multi-modal embedding fine-tuning.
        
        Validates:
        - Successful fine-tuning process
        - Performance metrics generation
        """
        # Create sample entities
        entities = [
            Entity(name="Quantum Mechanics", type=EntityType.CONCEPT),
            Entity(name="Warp Drive", type=EntityType.TECHNOLOGY)
        ]
        
        # Prepare training data
        training_data = {
            'text': {
                "Quantum Mechanics": "Quantum mechanics explores fundamental principles of nature",
                "Warp Drive": "Advanced propulsion technology for interstellar travel"
            },
            'images': {
                "Quantum Mechanics": ['tests/test_images/quantum_diagram.jpg'] if os.path.exists('tests/test_images/quantum_diagram.jpg') else [],
                "Warp Drive": ['tests/test_images/warp_drive_concept.jpg'] if os.path.exists('tests/test_images/warp_drive_concept.jpg') else []
            }
        }
        
        # Fine-tune embeddings
        metrics = await multi_modal_embedding_finetuner.fine_tune_embeddings(
            entities, training_data
        )
        
        # Validate metrics
        assert isinstance(metrics, dict), "Metrics should be a dictionary"
        assert 'text_embedding_dim' in metrics, "Missing text embedding dimension"
        assert 'image_embedding_dim' in metrics, "Missing image embedding dimension"
        assert 'loss' in metrics, "Missing loss metric"
        assert 'num_entities' in metrics, "Missing number of entities"
        assert 'modalities_fused' in metrics, "Missing modalities fusion flag"
    
    @pytest.mark.asyncio
    async def test_cross_modal_representation(self, multi_modal_embedding_finetuner):
        """
        Test cross-modal representation generation.
        
        Validates:
        - Successful cross-modal representation
        - Embedding vector properties
        """
        # Create sample entity
        entity = Entity(name="Quantum Mechanics", type=EntityType.CONCEPT)
        
        # Prepare inputs
        text = "Quantum mechanics explores fundamental principles of nature"
        image_path = 'tests/test_images/quantum_diagram.jpg' if os.path.exists('tests/test_images/quantum_diagram.jpg') else None
        
        # Generate cross-modal representation
        representation = await multi_modal_embedding_finetuner.generate_cross_modal_representation(
            entity, text, image_path
        )
        
        # Validate representation
        assert isinstance(representation, np.ndarray), "Representation should be a numpy array"
        assert representation.size == 768, "Unexpected representation size"
        assert not np.all(representation == 0), "Representation should not be zero vector"

class TestAdaptiveEmbeddingOptimizer:
    def test_initialization(self, adaptive_embedding_optimizer):
        """
        Test adaptive embedding optimizer initialization.
        
        Validates:
        - Successful initialization
        - Base model and tokenizer setup
        """
        # Validate base model
        assert hasattr(adaptive_embedding_optimizer, 'base_model'), "Missing base model"
        assert hasattr(adaptive_embedding_optimizer, 'tokenizer'), "Missing tokenizer"
        
        # Validate scheduler
        assert hasattr(adaptive_embedding_optimizer, 'scheduler'), "Missing learning rate scheduler"
    
    @pytest.mark.skip(reason="Transfer learning implementation pending")
    def test_transfer_learning(self, adaptive_embedding_optimizer):
        """
        Test transfer learning between domains.
        
        Validates:
        - Successful domain transfer
        - Performance metrics generation
        """
        # Create sample entities
        entities = [
            Entity(name="Quantum Mechanics", type=EntityType.CONCEPT),
            Entity(name="Warp Drive", type=EntityType.TECHNOLOGY)
        ]
        
        # Perform transfer learning
        metrics = adaptive_embedding_optimizer.transfer_learning(
            source_domain='physics', 
            target_domain='engineering', 
            entities=entities
        )
        
        # Validate metrics
        assert isinstance(metrics, dict), "Metrics should be a dictionary"
        assert 'transfer_performance' in metrics, "Missing transfer performance metric"
        assert 'domain_similarity' in metrics, "Missing domain similarity metric"

class TestMetaLearningEmbeddingAdapter:
    @pytest.mark.asyncio
    async def test_few_shot_adaptation(self, meta_learning_embedding_adapter):
        """
        Test few-shot learning adaptation.
        
        Validates:
        - Successful few-shot learning
        - Adaptation performance metrics
        """
        # Create sample entities for support and query sets
        support_set = [
            Entity(name="Quantum Mechanics", type=EntityType.CONCEPT),
            Entity(name="Warp Drive", type=EntityType.TECHNOLOGY),
            Entity(name="Superconducting Circuit", type=EntityType.EXPERIMENT)
        ]
        
        query_set = [
            Entity(name="Quantum Computing", type=EntityType.TECHNOLOGY)
        ]
        
        # Perform few-shot adaptation
        adaptation_metrics = meta_learning_embedding_adapter.few_shot_adaptation(
            support_set, query_set, num_shots=3
        )
        
        # Validate adaptation metrics
        assert isinstance(adaptation_metrics, dict), "Adaptation metrics should be a dictionary"
        
        # Check specific metrics
        assert 'support_loss' in adaptation_metrics, "Missing support loss"
        assert 'query_loss' in adaptation_metrics, "Missing query loss"
        assert 'adaptation_score' in adaptation_metrics, "Missing adaptation score"
        
        # Validate metric properties
        assert adaptation_metrics['adaptation_score'] >= 0, "Invalid adaptation score"
        assert adaptation_metrics['adaptation_score'] <= 1, "Invalid adaptation score"

class TestMultiModalTransformerEmbedding:
    def test_cross_modal_fusion(self, multi_modal_transformer_embedding):
        """
        Test cross-modal embedding fusion.
        
        Validates:
        - Successful embedding fusion
        - Dynamic modal weighting
        """
        # Create sample embeddings
        text_embedding = torch.randn(1, 768)  # SciBERT embedding size
        image_embedding = torch.randn(1, 2048)  # ResNet embedding size
        
        # Perform cross-modal fusion
        fused_embedding = multi_modal_transformer_embedding(
            text_embedding, image_embedding
        )
        
        # Validate fused embedding
        assert isinstance(fused_embedding, torch.Tensor), "Fused embedding should be a tensor"
        assert fused_embedding.size(1) == 768, "Unexpected fused embedding dimension"
        assert not torch.isnan(fused_embedding).any(), "Fused embedding contains NaN values"
    
    def test_dynamic_modal_weighting(self, multi_modal_transformer_embedding):
        """
        Test dynamic modal weighting mechanism.
        
        Validates:
        - Attention mechanism functionality
        - Modal importance adaptation
        """
        # Create multiple embedding pairs with varying characteristics
        text_embeddings = torch.randn(3, 768)
        image_embeddings = torch.randn(3, 2048)
        
        # Perform cross-modal fusion for multiple samples
        fused_embeddings = multi_modal_transformer_embedding(
            text_embeddings, image_embeddings
        )
        
        # Validate dynamic weighting
        assert fused_embeddings.size(0) == 3, "Incorrect number of fused embeddings"
        assert fused_embeddings.size(1) == 768, "Unexpected fused embedding dimension"

# Quantization and Pruning Tests
class TestEmbeddingModelOptimization:
    def test_model_quantization(self, multi_modal_embedding_finetuner):
        """
        Test embedding model quantization.
        
        Validates:
        - Successful model quantization
        - Reduced model size and computational complexity
        """
        # Quantize the embedding model
        quantized_model = quantize_embedding_model(
            multi_modal_embedding_finetuner.fusion_layer
        )
        
        # Validate quantization
        assert hasattr(quantized_model, 'qconfig'), "Quantization configuration missing"
        
        # Check computational type
        for module in quantized_model.modules():
            if hasattr(module, 'weight'):
                assert isinstance(module.weight, torch.nn.Parameter), "Weight quantization failed"
    
    def test_model_pruning(self, multi_modal_embedding_finetuner):
        """
        Test embedding model pruning.
        
        Validates:
        - Successful model pruning
        - Reduced model complexity
        """
        # Prune the embedding model
        pruned_model = prune_embedding_model(
            multi_modal_embedding_finetuner.fusion_layer,
            pruning_rate=0.3
        )
        
        # Validate pruning
        total_params = sum(p.numel() for p in pruned_model.parameters())
        pruned_params = sum(
            p.numel() for p in pruned_model.parameters() 
            if hasattr(p, 'mask') and p.mask is not None
        )
        
        # Check pruning effectiveness
        assert pruned_params > 0, "No parameters pruned"
        assert pruned_params / total_params >= 0.2, "Insufficient pruning"

# Performance and Scalability Tests
class TestAdvancedEmbeddingPerformance:
    @pytest.mark.parametrize("num_entities", [10, 50, 100])
    @pytest.mark.asyncio
    async def test_large_scale_meta_learning(
        self, 
        multi_modal_embedding_finetuner, 
        num_entities
    ):
        """
        Test meta-learning performance with large number of entities.
        
        Validates:
        - Ability to process multiple entities
        - Reasonable adaptation time
        """
        # Generate large set of entities
        entities = [
            Entity(
                name=f"Entity_{i}", 
                type=EntityType(i % len(EntityType) + 1)
            ) for i in range(num_entities)
        ]
        
        # Prepare training data
        training_data = {
            'text': {entity.name: f"Description for {entity.name}" for entity in entities},
            'images': {}  # No images for performance test
        }
        
        # Measure meta-learning performance
        import time
        start_time = time.time()
        
        metrics = await multi_modal_embedding_finetuner.fine_tune_embeddings(
            entities, training_data
        )
        
        adaptation_time = time.time() - start_time
        
        # Validate performance
        assert metrics is not None, "No metrics returned from large-scale meta-learning"
        assert adaptation_time < 30, f"Meta-learning adaptation took too long: {adaptation_time} seconds"
        
        # Check result structure
        assert 'adaptation_score' in metrics or 'loss' in metrics, "Missing adaptation metrics"

class TestPerformanceMetricsTracker:
    def test_task_performance_metrics(self, performance_tracker):
        """
        Test computation of task-specific performance metrics.
        
        Validates:
        - Precision, recall, and F1 score calculation
        - Classification report generation
        """
        # Generate synthetic classification data
        X, y_true = make_classification(
            n_samples=100, 
            n_features=20, 
            n_classes=3, 
            random_state=42
        )
        
        # Simulate model predictions
        y_pred = np.random.randint(0, 3, size=len(y_true))
        
        # Compute performance metrics
        metrics = performance_tracker.update_task_performance(y_true, y_pred)
        
        # Validate metrics
        assert 'precision' in metrics, "Missing precision metric"
        assert 'recall' in metrics, "Missing recall metric"
        assert 'f1_score' in metrics, "Missing F1 score metric"
        assert 'classification_report' in metrics, "Missing classification report"
        
        # Check metric ranges
        assert 0 <= metrics['precision'] <= 1, "Invalid precision value"
        assert 0 <= metrics['recall'] <= 1, "Invalid recall value"
        assert 0 <= metrics['f1_score'] <= 1, "Invalid F1 score"
    
    def test_computational_efficiency(
        self, 
        performance_tracker, 
        multi_modal_embedding_finetuner
    ):
        """
        Test computational efficiency measurement.
        
        Validates:
        - Latency measurement
        - Memory usage tracking
        """
        # Prepare sample entities
        entities = [
            Entity(name="Quantum Computer", type=EntityType.TECHNOLOGY),
            Entity(name="Neural Network", type=EntityType.CONCEPT)
        ]
        
        # Measure computational efficiency
        efficiency_metrics = performance_tracker.measure_computational_efficiency(
            multi_modal_embedding_finetuner.generate_entity_embedding,
            entities
        )
        
        # Validate efficiency metrics
        assert 'avg_latency_ms' in efficiency_metrics, "Missing latency metric"
        assert 'avg_memory_usage_bytes' in efficiency_metrics, "Missing memory usage metric"
        assert 'latency_std_dev' in efficiency_metrics, "Missing latency standard deviation"
        
        # Check metric characteristics
        assert efficiency_metrics['avg_latency_ms'] > 0, "Invalid latency measurement"
        assert efficiency_metrics['latency_std_dev'] >= 0, "Invalid latency standard deviation"
    
    def test_embedding_quality_evaluation(
        self, 
        performance_tracker
    ):
        """
        Test embedding quality assessment.
        
        Validates:
        - Embedding variance computation
        - Inter-embedding distance calculation
        - Supervised embedding evaluation
        """
        # Generate synthetic embeddings
        embeddings = torch.randn(50, 768)  # 50 samples, 768-dim embeddings
        labels = torch.randint(0, 5, (50,))  # 5 classes
        
        # Evaluate embedding quality
        quality_metrics = performance_tracker.evaluate_embedding_quality(
            embeddings, 
            labels
        )
        
        # Validate quality metrics
        assert 'embedding_variance' in quality_metrics, "Missing embedding variance"
        assert 'avg_inter_embedding_distance' in quality_metrics, "Missing inter-embedding distance"
        assert 'avg_intra_class_distance' in quality_metrics, "Missing intra-class distance"
        assert 'avg_inter_class_distance' in quality_metrics, "Missing inter-class distance"
        
        # Check metric characteristics
        assert quality_metrics['embedding_variance'] > 0, "Invalid embedding variance"
        assert quality_metrics['avg_inter_embedding_distance'] > 0, "Invalid inter-embedding distance"
    
    def test_modal_contribution_tracking(
        self, 
        performance_tracker
    ):
        """
        Test modal contribution analysis.
        
        Validates:
        - Modal similarity computation
        - Modal variance tracking
        """
        # Generate synthetic modal embeddings
        text_embeddings = torch.randn(10, 768)   # Text embeddings
        image_embeddings = torch.randn(10, 2048) # Image embeddings
        
        # Track modal contributions
        modal_metrics = performance_tracker.track_modal_contribution(
            text_embeddings, 
            image_embeddings
        )
        
        # Validate modal metrics
        assert 'modal_similarity' in modal_metrics, "Missing modal similarity"
        assert 'text_modal_variance' in modal_metrics, "Missing text modal variance"
        assert 'image_modal_variance' in modal_metrics, "Missing image modal variance"
        
        # Check metric characteristics
        assert -1 <= modal_metrics['modal_similarity'] <= 1, "Invalid modal similarity"
        assert modal_metrics['text_modal_variance'] > 0, "Invalid text modal variance"
        assert modal_metrics['image_modal_variance'] > 0, "Invalid image modal variance"
    
    def test_comprehensive_performance_report(
        self, 
        performance_tracker, 
        multi_modal_embedding_finetuner
    ):
        """
        Test generation of comprehensive performance report.
        
        Validates:
        - Report generation with multiple metric types
        - Aggregation of performance metrics
        """
        # Simulate multiple embedding fine-tuning runs
        entities = [
            Entity(name="Quantum Mechanics", type=EntityType.CONCEPT),
            Entity(name="Warp Drive", type=EntityType.TECHNOLOGY),
            Entity(name="Neural Network", type=EntityType.CONCEPT)
        ]
        
        # Perform multiple fine-tuning iterations
        for _ in range(5):
            metrics = multi_modal_embedding_finetuner.fine_tune_embeddings(
                entities, 
                {'text': {}, 'images': {}}
            )
        
        # Generate comprehensive report
        performance_report = performance_tracker.generate_comprehensive_report()
        
        # Validate report structure
        assert 'task_name' in performance_report, "Missing task name"
        assert 'performance_summary' in performance_report, "Missing performance summary"
        assert 'modal_contribution' in performance_report, "Missing modal contribution metrics"
        
        # Check performance summary metrics
        summary = performance_report['performance_summary']
        assert 'avg_precision' in summary, "Missing average precision"
        assert 'avg_recall' in summary, "Missing average recall"
        assert 'avg_f1_score' in summary, "Missing average F1 score"
        assert 'avg_latency_ms' in summary, "Missing average latency"
        assert 'embedding_variance' in summary, "Missing embedding variance"
        
        # Validate metric ranges
        assert 0 <= summary['avg_precision'] <= 1, "Invalid average precision"
        assert 0 <= summary['avg_recall'] <= 1, "Invalid average recall"
        assert 0 <= summary['avg_f1_score'] <= 1, "Invalid average F1 score"
        assert summary['avg_latency_ms'] > 0, "Invalid average latency"

class TestAdaptiveFinetuningStrategy:
    @pytest.fixture
    def adaptive_finetuning_strategy(self, multi_modal_embedding_finetuner):
        """
        Create an adaptive fine-tuning strategy for testing.
        """
        return AdaptiveFinetuningStrategy(
            base_model=multi_modal_embedding_finetuner.fusion_layer
        )
    
    def test_layer_wise_learning_rates(self, adaptive_finetuning_strategy):
        """
        Test layer-wise adaptive learning rate computation.
        
        Validates:
        - Learning rates computed for different layers
        - Learning rates within expected range
        """
        layer_lrs = adaptive_finetuning_strategy._compute_layer_learning_rates()
        
        # Validate layer learning rates
        assert len(layer_lrs) > 0, "No layer learning rates computed"
        
        for lr in layer_lrs:
            assert lr > 0, "Invalid learning rate"
            assert lr <= adaptive_finetuning_strategy.initial_lr, "Learning rate exceeds initial rate"
    
    def test_gradient_centralization(self, adaptive_finetuning_strategy):
        """
        Test gradient centralization mechanism.
        
        Validates:
        - Gradient centralization does not break model
        - Gradients are modified as expected
        """
        # Create a dummy loss tensor
        dummy_loss = torch.tensor(1.0, requires_grad=True)
        
        # Store original gradients
        original_gradients = {}
        for name, param in adaptive_finetuning_strategy.base_model.named_parameters():
            if param.requires_grad:
                original_gradients[name] = param.grad.clone() if param.grad is not None else None
        
        # Apply gradient centralization
        adaptive_finetuning_strategy.gradient_centralization(dummy_loss)
        
        # Validate gradient modifications
        for name, param in adaptive_finetuning_strategy.base_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                # Check that gradient has been modified
                assert not torch.equal(
                    param.grad, 
                    original_gradients.get(name, torch.zeros_like(param.grad))
                ), f"Gradient for {name} not centralized"
    
    def test_adaptive_regularization(self, adaptive_finetuning_strategy):
        """
        Test adaptive regularization loss computation.
        
        Validates:
        - Regularization loss computation
        - Loss reflects embedding characteristics
        """
        # Generate synthetic embeddings with multiple classes
        embeddings = torch.randn(50, 768)  # 50 samples, 768-dim
        labels = torch.randint(0, 5, (50,))  # 5 classes
        
        # Compute adaptive regularization loss
        reg_loss = adaptive_finetuning_strategy.adaptive_regularization(
            embeddings, 
            labels
        )
        
        # Validate regularization loss
        assert isinstance(reg_loss, torch.Tensor), "Invalid regularization loss type"
        assert reg_loss.requires_grad, "Regularization loss not differentiable"
        
        # Check loss magnitude
        assert not torch.isnan(reg_loss), "Regularization loss contains NaN"
    
    def test_adaptive_fine_tuning_process(
        self, 
        adaptive_finetuning_strategy,
        multi_modal_embedding_finetuner
    ):
        """
        Test complete adaptive fine-tuning process.
        
        Validates:
        - Successful fine-tuning execution
        - Performance metrics generation
        - Learning rate scheduling
        """
        # Prepare sample data
        entities = [
            Entity(name="Quantum Computer", type=EntityType.TECHNOLOGY),
            Entity(name="Neural Network", type=EntityType.CONCEPT),
            Entity(name="Warp Drive", type=EntityType.TECHNOLOGY)
        ]
        
        # Generate embeddings
        text_embeddings = torch.stack([
            multi_modal_embedding_finetuner._preprocess_text(entity.name) 
            for entity in entities
        ])
        labels = torch.tensor([entity.type.value for entity in entities])
        
        # Perform fine-tuning
        fine_tuning_metrics = adaptive_finetuning_strategy.fine_tune(
            text_embeddings, 
            labels, 
            num_epochs=3
        )
        
        # Validate fine-tuning metrics
        assert 'loss_history' in fine_tuning_metrics, "Missing loss history"
        assert 'lr_history' in fine_tuning_metrics, "Missing learning rate history"
        
        # Check loss history
        assert len(fine_tuning_metrics['loss_history']) == 3, "Incorrect number of epochs"
        for loss in fine_tuning_metrics['loss_history']:
            assert not np.isnan(loss), "NaN loss detected"
        
        # Check learning rate history
        assert len(fine_tuning_metrics['lr_history']) == 3, "Incorrect number of learning rates"
        for lr in fine_tuning_metrics['lr_history']:
            assert lr > 0, "Invalid learning rate"
            assert lr <= adaptive_finetuning_strategy.initial_lr, "Learning rate exceeds initial rate"
    
    @pytest.mark.parametrize("num_classes", [2, 5, 10])
    def test_scalability_and_robustness(
        self, 
        adaptive_finetuning_strategy, 
        num_classes
    ):
        """
        Test adaptive fine-tuning scalability and robustness.
        
        Validates:
        - Performance across different numbers of classes
        - Consistent fine-tuning behavior
        """
        # Generate synthetic multi-class embeddings
        embedding_dim = 768
        num_samples = 100
        
        embeddings = torch.randn(num_samples, embedding_dim)
        labels = torch.randint(0, num_classes, (num_samples,))
        
        # Perform fine-tuning
        fine_tuning_metrics = adaptive_finetuning_strategy.fine_tune(
            embeddings, 
            labels, 
            num_epochs=5
        )
        
        # Validate metrics consistency
        assert len(fine_tuning_metrics['loss_history']) == 5, "Incorrect loss history length"
        assert len(fine_tuning_metrics['lr_history']) == 5, "Incorrect learning rate history length"
        
        # Check loss convergence
        loss_history = fine_tuning_metrics['loss_history']
        assert all(not np.isnan(loss) for loss in loss_history), "NaN losses detected"
        
        # Optional: Basic convergence check
        if len(loss_history) > 1:
            assert loss_history[0] >= loss_history[-1], "Loss should generally decrease"
