"""
TFF Client Simulation
Simulates federated learning clients for TFF experiments
"""

import os
import sys
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import time

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import tensorflow_federated as tff
    TFF_AVAILABLE = True
except ImportError:
    TFF_AVAILABLE = False


@dataclass
class ClientConfig:
    """Configuration for a TFF client."""
    client_id: str
    batch_size: int = 32
    local_epochs: int = 1
    learning_rate: float = 0.001
    shuffle_buffer: int = 1000


class TFFClientSimulator:
    """
    Simulates TFF clients for federated learning experiments.
    
    In TFF, client simulation is handled by the framework itself,
    but this class provides utilities for monitoring and analysis.
    """
    
    def __init__(self, config: ClientConfig):
        """
        Initialize client simulator.
        
        Args:
            config: Client configuration
        """
        self.config = config
        self.local_model = None
        self.training_history = []
        
    def create_local_dataset(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> tf.data.Dataset:
        """
        Create a TF dataset for local training.
        
        Args:
            X: Feature array
            y: Label array
            
        Returns:
            TensorFlow Dataset
        """
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=self.config.shuffle_buffer)
        dataset = dataset.batch(self.config.batch_size)
        dataset = dataset.repeat(self.config.local_epochs)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def simulate_local_training(
        self,
        model: tf.keras.Model,
        dataset: tf.data.Dataset,
        verbose: int = 0
    ) -> Dict:
        """
        Simulate local training on client data.
        
        This is primarily for measuring local computation metrics.
        In actual TFF, training is handled by the framework.
        
        Args:
            model: Keras model
            dataset: Training dataset
            verbose: Verbosity level
            
        Returns:
            Training metrics
        """
        start_time = time.time()
        
        # Compile if needed
        if not model.optimizer:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(self.config.learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        
        # Train
        history = model.fit(dataset, verbose=verbose)
        
        training_time = time.time() - start_time
        
        metrics = {
            'client_id': self.config.client_id,
            'training_time': training_time,
            'final_loss': float(history.history['loss'][-1]),
            'final_accuracy': float(history.history['accuracy'][-1])
        }
        
        self.training_history.append(metrics)
        
        return metrics
    
    def get_model_update_size(self, model: tf.keras.Model) -> int:
        """
        Calculate the size of model updates in bytes.
        
        Args:
            model: Keras model
            
        Returns:
            Size in bytes
        """
        total_size = 0
        for weight in model.get_weights():
            total_size += weight.nbytes
        
        return total_size


class TFFClientFactory:
    """Factory for creating TFF client datasets."""
    
    def __init__(
        self,
        client_data: Dict[str, Dict[str, np.ndarray]],
        batch_size: int = 32,
        local_epochs: int = 1
    ):
        """
        Initialize client factory.
        
        Args:
            client_data: Dictionary mapping client IDs to data dicts
            batch_size: Batch size for all clients
            local_epochs: Number of local epochs
        """
        self.client_data = client_data
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.client_ids = list(client_data.keys())
        
    def create_tf_dataset_for_client(self, client_id: str) -> tf.data.Dataset:
        """
        Create TF dataset for a specific client.
        
        Args:
            client_id: Client identifier
            
        Returns:
            TensorFlow Dataset
        """
        if client_id not in self.client_data:
            raise ValueError(f"Unknown client: {client_id}")
        
        data = self.client_data[client_id]
        X, y = data['X'], data['y']
        
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        dataset = dataset.shuffle(buffer_size=len(y))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat(self.local_epochs)
        
        return dataset
    
    def get_federated_data(
        self, 
        client_ids: Optional[List[str]] = None
    ) -> List[tf.data.Dataset]:
        """
        Get federated data for selected clients.
        
        Args:
            client_ids: List of client IDs (None for all)
            
        Returns:
            List of TF Datasets
        """
        if client_ids is None:
            client_ids = self.client_ids
        
        return [
            self.create_tf_dataset_for_client(cid)
            for cid in client_ids
        ]
    
    def get_client_stats(self) -> Dict[str, Dict]:
        """Get statistics about client data distribution."""
        stats = {}
        
        for client_id, data in self.client_data.items():
            y = data['y']
            stats[client_id] = {
                'total_samples': len(y),
                'positive_ratio': float(np.mean(y)),
                'negative_ratio': float(1 - np.mean(y))
            }
        
        return stats


def create_tff_client_data_fn(
    client_data: Dict[str, Dict[str, np.ndarray]],
    batch_size: int = 32
) -> Callable[[str], tf.data.Dataset]:
    """
    Create a client data function for TFF.
    
    Args:
        client_data: Dictionary mapping client IDs to data
        batch_size: Batch size
        
    Returns:
        Function that returns dataset for a client ID
    """
    def client_data_fn(client_id: str) -> tf.data.Dataset:
        data = client_data[client_id]
        dataset = tf.data.Dataset.from_tensor_slices((data['X'], data['y']))
        dataset = dataset.shuffle(buffer_size=len(data['y']))
        dataset = dataset.batch(batch_size)
        return dataset
    
    return client_data_fn


if __name__ == "__main__":
    # Example usage
    print("TFF Client Simulation Example")
    print("=" * 50)
    
    # Create dummy data
    np.random.seed(42)
    client_data = {
        'client_0': {
            'X': np.random.randint(0, 1000, (100, 256)),
            'y': np.random.randint(0, 2, 100).astype(np.float32)
        },
        'client_1': {
            'X': np.random.randint(0, 1000, (80, 256)),
            'y': np.random.randint(0, 2, 80).astype(np.float32)
        }
    }
    
    # Create factory
    factory = TFFClientFactory(client_data, batch_size=16)
    
    # Get stats
    print("\nClient Statistics:")
    for client_id, stats in factory.get_client_stats().items():
        print(f"  {client_id}: {stats['total_samples']} samples, "
              f"{stats['positive_ratio']:.2%} positive")
    
    # Get federated data
    fed_data = factory.get_federated_data()
    print(f"\nCreated {len(fed_data)} client datasets")
    
    # Simulate local training
    config = ClientConfig(client_id='client_0')
    simulator = TFFClientSimulator(config)
    
    # Create simple model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(256,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Create dataset
    dataset = simulator.create_local_dataset(
        client_data['client_0']['X'],
        client_data['client_0']['y']
    )
    
    # Simulate training
    metrics = simulator.simulate_local_training(model, dataset)
    print(f"\nLocal training metrics: {metrics}")
