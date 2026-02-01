"""
Flower Client Implementation for Bug Prediction
Implements the Flower Client interface for FL training
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
import time

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Flower imports
try:
    import flwr as fl
    from flwr.common import (
        Code,
        EvaluateIns,
        EvaluateRes,
        FitIns,
        FitRes,
        GetParametersIns,
        GetParametersRes,
        Status,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    print("Flower not installed. Install with: pip install flwr")

from flower_model import (
    SimpleBugPredictor,
    get_model_parameters,
    set_model_parameters,
    get_model_size
)


class BugPredictionClient(fl.client.NumPyClient if FLOWER_AVAILABLE else object):
    """
    Flower client for bug prediction.
    
    Implements the NumPyClient interface for simpler parameter handling.
    """
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: Tuple[np.ndarray, np.ndarray],
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        batch_size: int = 32,
        local_epochs: int = 1,
        learning_rate: float = 0.001,
        device: str = 'cpu'
    ):
        """
        Initialize Flower client.
        
        Args:
            client_id: Unique client identifier
            model: PyTorch model
            train_data: Tuple of (X_train, y_train)
            test_data: Optional tuple of (X_test, y_test)
            batch_size: Training batch size
            local_epochs: Number of local training epochs
            learning_rate: Learning rate
            device: Device to use ('cpu' or 'cuda')
        """
        self.client_id = client_id
        self.model = model
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device
        
        # Move model to device
        self.model.to(self.device)
        
        # Create data loaders
        self.train_loader = self._create_dataloader(train_data, shuffle=True)
        self.test_loader = None
        if test_data is not None:
            self.test_loader = self._create_dataloader(test_data, shuffle=False)
        
        # Track metrics
        self.training_metrics = []
        
    def _create_dataloader(
        self, 
        data: Tuple[np.ndarray, np.ndarray],
        shuffle: bool = True
    ) -> DataLoader:
        """Create a PyTorch DataLoader."""
        X, y = data
        
        # Convert to tensors
        X_tensor = torch.LongTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle
        )
    
    def get_parameters(self, config: Dict = {}) -> List[np.ndarray]:
        """Get model parameters as numpy arrays."""
        return get_model_parameters(self.model)
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from numpy arrays."""
        set_model_parameters(self.model, parameters)
    
    def fit(
        self, 
        parameters: List[np.ndarray], 
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train the model on local data.
        
        Args:
            parameters: Global model parameters
            config: Configuration dictionary
            
        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        # Set global parameters
        self.set_parameters(parameters)
        
        # Get training config
        epochs = config.get('local_epochs', self.local_epochs)
        lr = config.get('learning_rate', self.learning_rate)
        
        # Setup training
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in self.train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_X.size(0)
                predictions = (outputs > 0.5).float()
                total_correct += (predictions == batch_y).sum().item()
                total_samples += batch_X.size(0)
            
            total_loss += epoch_loss
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        avg_loss = total_loss / (total_samples * epochs)
        accuracy = total_correct / (total_samples * epochs)
        
        # Calculate communication size
        param_bytes = sum(p.nbytes for p in self.get_parameters())
        
        metrics = {
            'client_id': self.client_id,
            'loss': float(avg_loss),
            'accuracy': float(accuracy),
            'training_time': training_time,
            'param_bytes': param_bytes
        }
        
        self.training_metrics.append(metrics)
        
        return self.get_parameters(), total_samples // epochs, metrics
    
    def evaluate(
        self, 
        parameters: List[np.ndarray], 
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate model on local test data.
        
        Args:
            parameters: Global model parameters
            config: Configuration dictionary
            
        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        if self.test_loader is None:
            return 0.0, 0, {}
        
        # Set parameters
        self.set_parameters(parameters)
        
        # Evaluation
        criterion = nn.BCELoss()
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_X, batch_y in self.test_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item() * batch_X.size(0)
                predictions = (outputs > 0.5).float()
                total_correct += (predictions == batch_y).sum().item()
                total_samples += batch_X.size(0)
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return float(avg_loss), total_samples, {'accuracy': float(accuracy)}


class BugPredictionClientRaw(fl.client.Client if FLOWER_AVAILABLE else object):
    """
    Lower-level Flower client implementation.
    Provides more control over the FL process.
    """
    
    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_data: Tuple[np.ndarray, np.ndarray],
        batch_size: int = 32,
        local_epochs: int = 1,
        device: str = 'cpu'
    ):
        """Initialize the raw client."""
        self.client_id = client_id
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        
        # Create dataset
        X, y = train_data
        self.train_dataset = TensorDataset(
            torch.LongTensor(X),
            torch.FloatTensor(y)
        )
        
    def get_parameters(self, ins: 'GetParametersIns') -> 'GetParametersRes':
        """Get model parameters."""
        parameters = ndarrays_to_parameters(get_model_parameters(self.model))
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters
        )
    
    def fit(self, ins: 'FitIns') -> 'FitRes':
        """Train the model."""
        # Get parameters from server
        parameters = parameters_to_ndarrays(ins.parameters)
        set_model_parameters(self.model, parameters)
        
        # Train
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for _ in range(self.local_epochs):
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        # Return updated parameters
        updated_parameters = ndarrays_to_parameters(get_model_parameters(self.model))
        
        return FitRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=updated_parameters,
            num_examples=len(self.train_dataset),
            metrics={}
        )
    
    def evaluate(self, ins: 'EvaluateIns') -> 'EvaluateRes':
        """Evaluate the model."""
        parameters = parameters_to_ndarrays(ins.parameters)
        set_model_parameters(self.model, parameters)
        
        # Simple evaluation (would use test data in practice)
        return EvaluateRes(
            status=Status(code=Code.OK, message="Success"),
            loss=0.0,
            num_examples=len(self.train_dataset),
            metrics={}
        )


def create_client_fn(
    client_data: Dict[str, Dict[str, np.ndarray]],
    model_fn: callable,
    batch_size: int = 32,
    local_epochs: int = 1,
    device: str = 'cpu'
):
    """
    Create a client function for Flower simulation.
    
    Args:
        client_data: Dictionary mapping client IDs to data
        model_fn: Function that creates a new model
        batch_size: Training batch size
        local_epochs: Local epochs per round
        device: Device to use
        
    Returns:
        Function that creates clients
    """
    def client_fn(cid: str) -> BugPredictionClient:
        """Create a client with the given ID."""
        data = client_data[cid]
        model = model_fn()
        
        return BugPredictionClient(
            client_id=cid,
            model=model,
            train_data=(data['X'], data['y']),
            batch_size=batch_size,
            local_epochs=local_epochs,
            device=device
        )
    
    return client_fn


if __name__ == "__main__":
    # Test client implementation
    print("Testing Flower Client Implementation")
    print("=" * 50)
    
    if not FLOWER_AVAILABLE:
        print("\nFlower not installed. Cannot run test.")
        sys.exit(0)
    
    # Create dummy data
    np.random.seed(42)
    vocab_size = 1000
    seq_length = 128
    num_samples = 100
    
    X_train = np.random.randint(0, vocab_size, (num_samples, seq_length))
    y_train = np.random.randint(0, 2, num_samples).astype(np.float32)
    
    # Create model
    model = SimpleBugPredictor(vocab_size)
    
    # Create client
    client = BugPredictionClient(
        client_id='test_client',
        model=model,
        train_data=(X_train, y_train),
        batch_size=16,
        local_epochs=2
    )
    
    # Test get_parameters
    print("\n1. Testing get_parameters...")
    params = client.get_parameters()
    print(f"   Number of parameter arrays: {len(params)}")
    print(f"   Total bytes: {sum(p.nbytes for p in params):,}")
    
    # Test fit
    print("\n2. Testing fit...")
    updated_params, num_samples, metrics = client.fit(params, {})
    print(f"   Trained on {num_samples} samples")
    print(f"   Metrics: {metrics}")
    
    # Test with simulation
    print("\n3. Client ready for Flower simulation")
