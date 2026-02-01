"""
Flower Server Implementation for Bug Prediction
Implements server-side logic for FL aggregation
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import time

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Flower imports
try:
    import flwr as fl
    from flwr.server.strategy import FedAvg, FedProx, FedAdam
    from flwr.common import (
        Metrics,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    FLOWER_AVAILABLE = True
except ImportError:
    FLOWER_AVAILABLE = False
    print("Flower not installed. Install with: pip install flwr")


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Weighted average of metrics across clients.
    
    Args:
        metrics: List of (num_examples, metrics_dict) tuples
        
    Returns:
        Aggregated metrics
    """
    if not metrics:
        return {}
    
    # Extract values
    accuracies = [m['accuracy'] for _, m in metrics if 'accuracy' in m]
    examples = [num for num, m in metrics if 'accuracy' in m]
    
    if not accuracies:
        return {}
    
    # Weighted average
    weighted_acc = sum(a * n for a, n in zip(accuracies, examples)) / sum(examples)
    
    return {'accuracy': weighted_acc}


class CustomFedAvg(FedAvg if FLOWER_AVAILABLE else object):
    """
    Custom FedAvg strategy with additional metrics tracking.
    """
    
    def __init__(
        self,
        *args,
        track_communication: bool = True,
        **kwargs
    ):
        """
        Initialize custom FedAvg.
        
        Args:
            track_communication: Whether to track communication costs
            *args, **kwargs: Arguments for FedAvg
        """
        if FLOWER_AVAILABLE:
            super().__init__(*args, **kwargs)
        
        self.track_communication = track_communication
        self.round_communication = []
        self.round_times = []
        self.round_metrics = []
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List,
        failures: List,
    ):
        """Aggregate fit results with tracking."""
        start_time = time.time()
        
        # Track communication
        if self.track_communication and results:
            bytes_sent = sum(
                sum(p.nbytes for p in parameters_to_ndarrays(fit_res.parameters))
                for _, fit_res in results
            )
            self.round_communication.append({
                'round': server_round,
                'bytes_received': bytes_sent,
                'num_clients': len(results),
                'num_failures': len(failures)
            })
        
        # Perform aggregation
        aggregated = super().aggregate_fit(server_round, results, failures)
        
        # Track time
        aggregation_time = time.time() - start_time
        self.round_times.append({
            'round': server_round,
            'aggregation_time': aggregation_time
        })
        
        return aggregated
    
    def get_communication_stats(self) -> Dict:
        """Get communication statistics."""
        if not self.round_communication:
            return {'total_bytes': 0, 'avg_bytes_per_round': 0}
        
        total_bytes = sum(r['bytes_received'] for r in self.round_communication)
        avg_bytes = total_bytes / len(self.round_communication)
        
        return {
            'total_bytes': total_bytes,
            'avg_bytes_per_round': avg_bytes,
            'rounds': self.round_communication
        }


class SecureFedAvg(FedAvg if FLOWER_AVAILABLE else object):
    """
    FedAvg with simulated secure aggregation.
    Demonstrates security overhead measurement.
    """
    
    def __init__(
        self,
        *args,
        encryption_overhead_ms: float = 5.0,
        **kwargs
    ):
        """
        Initialize secure FedAvg.
        
        Args:
            encryption_overhead_ms: Simulated encryption overhead per client
            *args, **kwargs: Arguments for FedAvg
        """
        if FLOWER_AVAILABLE:
            super().__init__(*args, **kwargs)
        
        self.encryption_overhead_ms = encryption_overhead_ms
        self.security_overhead = []
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List,
        failures: List,
    ):
        """Aggregate with simulated encryption overhead."""
        # Simulate encryption/decryption overhead
        num_clients = len(results)
        overhead_ms = num_clients * self.encryption_overhead_ms * 2  # encrypt + decrypt
        time.sleep(overhead_ms / 1000)
        
        self.security_overhead.append({
            'round': server_round,
            'overhead_ms': overhead_ms,
            'num_clients': num_clients
        })
        
        return super().aggregate_fit(server_round, results, failures)
    
    def get_security_stats(self) -> Dict:
        """Get security overhead statistics."""
        if not self.security_overhead:
            return {'total_overhead_ms': 0}
        
        total_overhead = sum(r['overhead_ms'] for r in self.security_overhead)
        
        return {
            'total_overhead_ms': total_overhead,
            'avg_overhead_per_round': total_overhead / len(self.security_overhead),
            'rounds': self.security_overhead
        }


def create_server_strategy(
    strategy_name: str = 'fedavg',
    fraction_fit: float = 1.0,
    fraction_evaluate: float = 0.5,
    min_fit_clients: int = 2,
    min_evaluate_clients: int = 2,
    min_available_clients: int = 2,
    initial_parameters = None,
    **kwargs
):
    """
    Create a Flower server strategy.
    
    Args:
        strategy_name: Name of strategy ('fedavg', 'fedprox', 'fedadam', 'secure')
        fraction_fit: Fraction of clients for training
        fraction_evaluate: Fraction of clients for evaluation
        min_fit_clients: Minimum clients for training
        min_evaluate_clients: Minimum clients for evaluation
        min_available_clients: Minimum available clients
        initial_parameters: Initial model parameters
        **kwargs: Additional strategy-specific arguments
        
    Returns:
        Flower Strategy instance
    """
    if not FLOWER_AVAILABLE:
        raise RuntimeError("Flower not installed")
    
    common_args = {
        'fraction_fit': fraction_fit,
        'fraction_evaluate': fraction_evaluate,
        'min_fit_clients': min_fit_clients,
        'min_evaluate_clients': min_evaluate_clients,
        'min_available_clients': min_available_clients,
        'evaluate_metrics_aggregation_fn': weighted_average,
    }
    
    if initial_parameters is not None:
        common_args['initial_parameters'] = ndarrays_to_parameters(initial_parameters)
    
    if strategy_name == 'fedavg':
        return CustomFedAvg(**common_args, **kwargs)
    
    elif strategy_name == 'fedprox':
        proximal_mu = kwargs.pop('proximal_mu', 0.1)
        return FedProx(proximal_mu=proximal_mu, **common_args)
    
    elif strategy_name == 'fedadam':
        return FedAdam(**common_args)
    
    elif strategy_name == 'secure':
        encryption_overhead = kwargs.pop('encryption_overhead_ms', 5.0)
        return SecureFedAvg(
            encryption_overhead_ms=encryption_overhead,
            **common_args
        )
    
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def run_flower_server(
    strategy,
    num_rounds: int = 10,
    server_address: str = '[::]:8080'
) -> None:
    """
    Run Flower server.
    
    Args:
        strategy: Server strategy
        num_rounds: Number of FL rounds
        server_address: Server address
    """
    if not FLOWER_AVAILABLE:
        raise RuntimeError("Flower not installed")
    
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )


class FlowerServerConfig:
    """Configuration container for Flower server."""
    
    def __init__(
        self,
        num_rounds: int = 10,
        strategy_name: str = 'fedavg',
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.5,
        min_clients: int = 2,
        **kwargs
    ):
        """Initialize server configuration."""
        self.num_rounds = num_rounds
        self.strategy_name = strategy_name
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_clients = min_clients
        self.extra_args = kwargs
        
    def create_strategy(self, initial_parameters=None):
        """Create strategy from configuration."""
        return create_server_strategy(
            strategy_name=self.strategy_name,
            fraction_fit=self.fraction_fit,
            fraction_evaluate=self.fraction_evaluate,
            min_fit_clients=self.min_clients,
            min_evaluate_clients=self.min_clients,
            min_available_clients=self.min_clients,
            initial_parameters=initial_parameters,
            **self.extra_args
        )


if __name__ == "__main__":
    # Test server components
    print("Testing Flower Server Components")
    print("=" * 50)
    
    if not FLOWER_AVAILABLE:
        print("\nFlower not installed. Cannot run test.")
        sys.exit(0)
    
    # Test strategy creation
    print("\n1. Testing strategy creation...")
    
    strategies = ['fedavg', 'secure']
    for name in strategies:
        try:
            strategy = create_server_strategy(name)
            print(f"   ✓ {name}: Created successfully")
        except Exception as e:
            print(f"   ✗ {name}: {e}")
    
    # Test custom FedAvg
    print("\n2. Testing CustomFedAvg tracking...")
    custom_strategy = CustomFedAvg(track_communication=True)
    print(f"   Communication tracking: {custom_strategy.track_communication}")
    
    # Test SecureFedAvg
    print("\n3. Testing SecureFedAvg...")
    secure_strategy = SecureFedAvg(encryption_overhead_ms=10.0)
    print(f"   Encryption overhead: {secure_strategy.encryption_overhead_ms}ms")
    
    # Test server config
    print("\n4. Testing FlowerServerConfig...")
    config = FlowerServerConfig(
        num_rounds=5,
        strategy_name='fedavg',
        min_clients=3
    )
    print(f"   Rounds: {config.num_rounds}")
    print(f"   Strategy: {config.strategy_name}")
    print(f"   Min clients: {config.min_clients}")
    
    print("\n✓ All server components working")
