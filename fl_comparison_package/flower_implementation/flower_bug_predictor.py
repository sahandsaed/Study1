"""
Flower Federated Learning Implementation for Bug Prediction
Main training script using Flower's simulation mode
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Flower imports
try:
    import flwr as fl
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, Metrics
    from flwr.simulation import start_simulation
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    print("Flower not installed. Install with: pip install flwr[simulation]")

from utils.data_processor import BugPredictionDataProcessor
from utils.code_tokenizer import CodeTokenizer, CodeFeatureExtractor
from utils.metrics_collector import MetricsCollector

from flower_model import (
    SimpleBugPredictor,
    FeatureOnlyPredictor,
    get_model_parameters,
    set_model_parameters,
    get_model_size
)
from flower_client import BugPredictionClient, create_client_fn
from flower_server import CustomFedAvg, SecureFedAvg


class FlowerBugPredictor:
    """
    Flower implementation for bug prediction.
    
    This class demonstrates FL training using Flower framework,
    measuring flexibility and technical metrics for comparison.
    """
    
    def __init__(
        self,
        data_path: str,
        num_clients: int = 5,
        num_rounds: int = 10,
        local_epochs: int = 1,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        max_length: int = 256,
        vocab_size: int = 5000,
        model_type: str = 'simple',
        device: str = 'cpu'
    ):
        """
        Initialize Flower Bug Predictor.
        
        Args:
            data_path: Path to the bug prediction dataset
            num_clients: Number of FL clients to simulate
            num_rounds: Number of FL training rounds
            local_epochs: Number of local training epochs per round
            batch_size: Batch size for training
            learning_rate: Learning rate
            max_length: Maximum sequence length
            vocab_size: Vocabulary size for tokenizer
            model_type: 'simple' or 'feature_only'
            device: Device to use ('cpu' or 'cuda')
        """
        self.data_path = data_path
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.device = device
        
        # Initialize components
        self.tokenizer = CodeTokenizer(max_length=max_length, vocab_size=vocab_size)
        self.feature_extractor = CodeFeatureExtractor()
        self.metrics_collector = MetricsCollector('flower', 'bug_prediction')
        
        # Will be initialized later
        self.client_data = None
        self.test_data = None
        self.model = None
        self.strategy = None
        
    def _create_model(self) -> nn.Module:
        """Create the PyTorch model."""
        if self.model_type == 'simple':
            model = SimpleBugPredictor(len(self.tokenizer.vocab))
        else:  # feature_only
            num_features = len(self.feature_extractor.feature_names)
            model = FeatureOnlyPredictor(num_features)
        
        return model
    
    def load_and_preprocess_data(self) -> None:
        """Load and preprocess the bug prediction dataset."""
        print("Loading and preprocessing data...")
        
        # Load data
        processor = BugPredictionDataProcessor(self.data_path)
        processor.load_data()
        
        # Prepare binary classification data
        code_samples, labels = processor.prepare_binary_classification()
        
        # Fit tokenizer
        self.tokenizer.fit(code_samples)
        
        # Partition data for FL
        partitioned = processor.partition_iid(
            num_clients=self.num_clients,
            test_size=0.2
        )
        
        # Process client data
        self.client_data = {}
        for client_id, data in partitioned['clients'].items():
            if self.model_type == 'simple':
                X = self.tokenizer.encode_batch(data['X'])
            else:
                X = self.feature_extractor.extract_batch(data['X'])
            
            y = np.array(data['y'], dtype=np.float32)
            self.client_data[client_id] = {'X': X, 'y': y}
        
        # Process test data
        if self.model_type == 'simple':
            test_X = self.tokenizer.encode_batch(partitioned['test']['X'])
        else:
            test_X = self.feature_extractor.extract_batch(partitioned['test']['X'])
        
        test_y = np.array(partitioned['test']['y'], dtype=np.float32)
        self.test_data = {'X': test_X, 'y': test_y}
        
        print(f"Data loaded: {len(self.client_data)} clients, "
              f"{len(self.test_data['y'])} test samples")
    
    def setup_fl_process(self, secure: bool = False) -> None:
        """
        Set up the federated learning process.
        
        Args:
            secure: Whether to use secure aggregation simulation
        """
        if not FLOWER_AVAILABLE:
            raise RuntimeError("Flower is not installed")
        
        print("Setting up FL process...")
        self.metrics_collector.start_setup()
        
        # Create initial model
        self.model = self._create_model()
        initial_parameters = get_model_parameters(self.model)
        
        # Create strategy
        def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Metrics]]) -> Metrics:
            """Aggregate fit metrics."""
            if not metrics:
                return {}
            
            accuracies = [m.get('accuracy', 0) for _, m in metrics]
            losses = [m.get('loss', 0) for _, m in metrics]
            examples = [n for n, _ in metrics]
            
            total = sum(examples)
            if total == 0:
                return {}
            
            return {
                'accuracy': sum(a * n for a, n in zip(accuracies, examples)) / total,
                'loss': sum(l * n for l, n in zip(losses, examples)) / total,
            }
        
        if secure:
            self.strategy = SecureFedAvg(
                fraction_fit=1.0,
                fraction_evaluate=0.5,
                min_fit_clients=self.num_clients,
                min_evaluate_clients=max(1, self.num_clients // 2),
                min_available_clients=self.num_clients,
                initial_parameters=ndarrays_to_parameters(initial_parameters),
                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                encryption_overhead_ms=5.0
            )
        else:
            self.strategy = CustomFedAvg(
                fraction_fit=1.0,
                fraction_evaluate=0.5,
                min_fit_clients=self.num_clients,
                min_evaluate_clients=max(1, self.num_clients // 2),
                min_available_clients=self.num_clients,
                initial_parameters=ndarrays_to_parameters(initial_parameters),
                fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                track_communication=True
            )
        
        self.metrics_collector.end_setup()
        print(f"FL process setup complete (setup time: "
              f"{self.metrics_collector.experiment_metrics.setup_time_seconds:.2f}s)")
    
    def _client_fn(self, cid: str) -> BugPredictionClient:
        """Create a client with the given ID."""
        data = self.client_data[cid]
        model = self._create_model()
        
        return BugPredictionClient(
            client_id=cid,
            model=model,
            train_data=(data['X'], data['y']),
            batch_size=self.batch_size,
            local_epochs=self.local_epochs,
            learning_rate=self.learning_rate,
            device=self.device
        )
    
    def train(self) -> Dict:
        """
        Run federated learning training using Flower simulation.
        
        Returns:
            Dictionary with training results and metrics
        """
        print(f"\nStarting Flower training: {self.num_rounds} rounds, "
              f"{self.num_clients} clients")
        
        self.metrics_collector.start_experiment(
            num_clients=self.num_clients,
            num_rounds=self.num_rounds,
            local_epochs=self.local_epochs,
            batch_size=self.batch_size,
            model_type=self.model_type
        )
        
        # Record model size
        model_size = get_model_size(self.model)
        self.metrics_collector.record_model_size(model_size)
        
        # Run simulation
        client_ids = list(self.client_data.keys())
        
        # Custom training loop for better metrics tracking
        training_history = []
        global_model = self._create_model()
        set_model_parameters(global_model, get_model_parameters(self.model))
        
        for round_num in range(1, self.num_rounds + 1):
            self.metrics_collector.start_round(round_num)
            round_start = time.time()
            
            # Simulate client training
            client_updates = []
            client_samples = []
            
            for cid in client_ids:
                # Create client
                client = self._client_fn(cid)
                
                # Get current global parameters
                global_params = get_model_parameters(global_model)
                
                # Train locally
                updated_params, num_samples, metrics = client.fit(
                    global_params,
                    {'local_epochs': self.local_epochs}
                )
                
                client_updates.append(updated_params)
                client_samples.append(num_samples)
            
            # Aggregate updates (FedAvg)
            total_samples = sum(client_samples)
            aggregated_params = []
            
            for param_idx in range(len(client_updates[0])):
                weighted_sum = np.zeros_like(client_updates[0][param_idx])
                for client_params, num_samples in zip(client_updates, client_samples):
                    weighted_sum += client_params[param_idx] * (num_samples / total_samples)
                aggregated_params.append(weighted_sum)
            
            # Update global model
            set_model_parameters(global_model, aggregated_params)
            
            round_time = time.time() - round_start
            
            # Evaluate
            accuracy, loss = self._evaluate_model(global_model)
            
            # Record communication
            comm_bytes = model_size * 2 * self.num_clients
            self.metrics_collector.record_communication(
                sent=comm_bytes // 2,
                received=comm_bytes // 2
            )
            
            # Record round metrics
            self.metrics_collector.end_round(
                train_loss=loss,
                train_accuracy=accuracy,
                num_clients=self.num_clients
            )
            
            training_history.append({
                'round': round_num,
                'loss': loss,
                'accuracy': accuracy,
                'time': round_time
            })
        
        # Final evaluation
        final_accuracy, final_loss = self._evaluate_model(global_model)
        
        # End experiment
        experiment_metrics = self.metrics_collector.end_experiment(
            final_accuracy=final_accuracy,
            final_loss=final_loss
        )
        
        # Store final model
        self.model = global_model
        
        return {
            'training_history': training_history,
            'final_accuracy': final_accuracy,
            'final_loss': final_loss,
            'metrics': experiment_metrics
        }
    
    def _evaluate_model(self, model: nn.Module) -> Tuple[float, float]:
        """Evaluate model on test data."""
        model.eval()
        model.to(self.device)
        
        X_test = torch.LongTensor(self.test_data['X']).to(self.device)
        y_test = torch.FloatTensor(self.test_data['y']).to(self.device)
        
        criterion = nn.BCELoss()
        
        with torch.no_grad():
            outputs = model(X_test)
            loss = criterion(outputs, y_test)
            
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y_test).float().mean()
        
        return float(accuracy), float(loss)
    
    def save_results(self, output_dir: str) -> None:
        """Save experiment results."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        self.metrics_collector.save_results(
            os.path.join(output_dir, 'flower_metrics.json')
        )
        
        # Save model
        torch.save(
            self.model.state_dict(),
            os.path.join(output_dir, 'flower_model.pt')
        )
        
        print(f"Results saved to {output_dir}")


def run_flower_experiment(
    data_path: str,
    output_dir: str = 'results',
    num_clients: int = 5,
    num_rounds: int = 10,
    secure: bool = False,
    **kwargs
) -> Dict:
    """
    Run a complete Flower experiment.
    
    Args:
        data_path: Path to dataset
        output_dir: Directory for results
        num_clients: Number of clients
        num_rounds: Number of rounds
        secure: Use secure aggregation simulation
        **kwargs: Additional arguments for FlowerBugPredictor
        
    Returns:
        Experiment results
    """
    predictor = FlowerBugPredictor(
        data_path=data_path,
        num_clients=num_clients,
        num_rounds=num_rounds,
        **kwargs
    )
    
    # Load data
    predictor.load_and_preprocess_data()
    
    # Setup FL
    predictor.setup_fl_process(secure=secure)
    
    # Train
    results = predictor.train()
    
    # Save results
    predictor.save_results(output_dir)
    
    return results


# ============================================================================
# LINES OF CODE MEASUREMENT FOR FLEXIBILITY COMPARISON
# ============================================================================
# This implementation requires approximately:
# - Setup code: ~30 lines
# - Model definition: ~40 lines (in flower_model.py)
# - Client implementation: ~80 lines (in flower_client.py)
# - Server implementation: ~40 lines (in flower_server.py)
# - Data preprocessing: ~40 lines
# - Training loop: ~80 lines
# - Evaluation: ~20 lines
# Total: ~330 lines of implementation code
#
# Key Flower-specific requirements:
# 1. Client must implement NumPyClient or Client interface
# 2. Model parameters serialization handled by user
# 3. Strategy selection is flexible (FedAvg, FedProx, etc.)
# 4. Server/client can run separately or in simulation
# ============================================================================


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("Flower Federated Learning Bug Prediction")
    print("=" * 60)
    
    # Check if Flower is available
    if not FLOWER_AVAILABLE:
        print("\nFlower not installed. To install:")
        print("  pip install flwr[simulation]")
        print("\nThis script demonstrates the Flower implementation structure.")
        sys.exit(0)
    
    # Run experiment
    results = run_flower_experiment(
        data_path='data/dataset_pairs_1_.json',
        output_dir='results/flower',
        num_clients=5,
        num_rounds=10,
        local_epochs=1,
        batch_size=32,
        model_type='simple',
        secure=False
    )
    
    print("\n" + "=" * 60)
    print("Experiment Complete")
    print("=" * 60)
    print(f"Final Accuracy: {results['final_accuracy']:.4f}")
    print(f"Final Loss: {results['final_loss']:.4f}")
