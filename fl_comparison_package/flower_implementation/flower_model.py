"""
PyTorch Model for Bug Prediction
Used in Flower Federated Learning implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np


class SimpleBugPredictor(nn.Module):
    """
    Simple bug prediction model using embeddings and dense layers.
    Designed for Flower FL experiments.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        """
        Initialize model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super(SimpleBugPredictor, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # Global average pooling
        pooled = torch.mean(embedded, dim=1)  # (batch_size, embedding_dim)
        
        # Dense layers
        x = F.relu(self.fc1(pooled))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x.squeeze(-1)


class LSTMBugPredictor(nn.Module):
    """
    LSTM-based bug prediction model.
    More powerful but also more resource-intensive.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize LSTM model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            bidirectional: Use bidirectional LSTM
            dropout: Dropout rate
        """
        super(LSTMBugPredictor, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc1 = nn.Linear(lstm_output_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use last hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Dense layers
        x = F.relu(self.fc1(hidden))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x.squeeze(-1)


class FeatureOnlyPredictor(nn.Module):
    """
    Bug prediction using only handcrafted features.
    Useful as a baseline model.
    """
    
    def __init__(
        self,
        num_features: int = 20,
        hidden_dims: List[int] = [64, 32, 16],
        dropout: float = 0.3
    ):
        """
        Initialize feature-based model.
        
        Args:
            num_features: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super(FeatureOnlyPredictor, self).__init__()
        
        layers = []
        in_dim = num_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x).squeeze(-1)


class HybridBugPredictor(nn.Module):
    """
    Hybrid model combining code embeddings and features.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        num_features: int = 20,
        hidden_dim: int = 64,
        dropout: float = 0.3
    ):
        """
        Initialize hybrid model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            num_features: Number of handcrafted features
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super(HybridBugPredictor, self).__init__()
        
        # Token branch
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.token_fc = nn.Linear(embedding_dim, hidden_dim)
        
        # Feature branch
        self.feature_fc = nn.Linear(num_features, hidden_dim // 2)
        
        # Combined layers
        combined_dim = hidden_dim + hidden_dim // 2
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        tokens: torch.Tensor, 
        features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with both inputs."""
        # Token branch
        embedded = self.embedding(tokens)
        pooled = torch.mean(embedded, dim=1)
        token_out = F.relu(self.token_fc(pooled))
        token_out = self.dropout(token_out)
        
        # Feature branch
        feature_out = F.relu(self.feature_fc(features))
        feature_out = self.dropout(feature_out)
        
        # Combine
        combined = torch.cat([token_out, feature_out], dim=1)
        
        # Final layers
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        
        return x.squeeze(-1)


def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """
    Get model parameters as numpy arrays.
    Used for Flower communication.
    
    Args:
        model: PyTorch model
        
    Returns:
        List of parameter arrays
    """
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """
    Set model parameters from numpy arrays.
    Used for Flower communication.
    
    Args:
        model: PyTorch model
        parameters: List of parameter arrays
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def get_model_size(model: nn.Module) -> int:
    """
    Calculate model size in bytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Size in bytes
    """
    total_size = 0
    for param in model.parameters():
        total_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        total_size += buffer.nelement() * buffer.element_size()
    
    return total_size


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    print("Testing PyTorch models for Flower...")
    
    vocab_size = 5000
    batch_size = 4
    seq_length = 256
    num_features = 20
    
    # Create dummy input
    token_input = torch.randint(0, vocab_size, (batch_size, seq_length))
    feature_input = torch.randn(batch_size, num_features)
    
    # Test simple model
    print("\n1. Simple Bug Predictor:")
    simple_model = SimpleBugPredictor(vocab_size)
    output = simple_model(token_input)
    print(f"   Parameters: {count_parameters(simple_model):,}")
    print(f"   Size: {get_model_size(simple_model) / 1024:.2f} KB")
    print(f"   Output shape: {output.shape}")
    
    # Test LSTM model
    print("\n2. LSTM Bug Predictor:")
    lstm_model = LSTMBugPredictor(vocab_size)
    output = lstm_model(token_input)
    print(f"   Parameters: {count_parameters(lstm_model):,}")
    print(f"   Size: {get_model_size(lstm_model) / 1024:.2f} KB")
    print(f"   Output shape: {output.shape}")
    
    # Test feature-only model
    print("\n3. Feature-Only Predictor:")
    feature_model = FeatureOnlyPredictor(num_features)
    output = feature_model(feature_input)
    print(f"   Parameters: {count_parameters(feature_model):,}")
    print(f"   Size: {get_model_size(feature_model) / 1024:.2f} KB")
    print(f"   Output shape: {output.shape}")
    
    # Test hybrid model
    print("\n4. Hybrid Bug Predictor:")
    hybrid_model = HybridBugPredictor(vocab_size, num_features=num_features)
    output = hybrid_model(token_input, feature_input)
    print(f"   Parameters: {count_parameters(hybrid_model):,}")
    print(f"   Size: {get_model_size(hybrid_model) / 1024:.2f} KB")
    print(f"   Output shape: {output.shape}")
    
    # Test parameter extraction
    print("\n5. Parameter Extraction Test:")
    params = get_model_parameters(simple_model)
    print(f"   Number of parameter arrays: {len(params)}")
    print(f"   Total bytes: {sum(p.nbytes for p in params):,}")
