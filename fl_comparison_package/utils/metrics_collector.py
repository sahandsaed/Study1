"""
Metrics Collector for FL Framework Comparison
Collects performance, memory, communication, and other metrics
"""

import time
import psutil
import os
import sys
import json
import tracemalloc
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import threading
import numpy as np


@dataclass
class RoundMetrics:
    """Metrics for a single FL round."""
    round_number: int
    duration_seconds: float
    communication_bytes_sent: int = 0
    communication_bytes_received: int = 0
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    cpu_percent: float = 0.0
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    val_loss: float = 0.0
    val_accuracy: float = 0.0
    num_clients_participated: int = 0
    aggregation_time_seconds: float = 0.0


@dataclass
class ExperimentMetrics:
    """Complete metrics for an FL experiment."""
    framework: str
    experiment_name: str
    start_time: str = ""
    end_time: str = ""
    total_duration_seconds: float = 0.0
    num_rounds: int = 0
    num_clients: int = 0
    model_size_bytes: int = 0
    total_communication_bytes: int = 0
    peak_memory_mb: float = 0.0
    final_accuracy: float = 0.0
    final_loss: float = 0.0
    rounds: List[RoundMetrics] = field(default_factory=list)
    setup_time_seconds: float = 0.0
    teardown_time_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryMonitor:
    """Monitor memory usage in a separate thread."""
    
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.measurements = []
        self._stop_event = threading.Event()
        self._thread = None
        
    def _monitor(self):
        """Background monitoring function."""
        process = psutil.Process(os.getpid())
        while not self._stop_event.is_set():
            try:
                mem_info = process.memory_info()
                self.measurements.append(mem_info.rss / (1024 * 1024))  # MB
            except:
                pass
            time.sleep(self.interval)
    
    def start(self):
        """Start monitoring."""
        self.measurements = []
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        
    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return statistics."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        
        if not self.measurements:
            return {'peak': 0, 'avg': 0, 'min': 0}
            
        return {
            'peak': max(self.measurements),
            'avg': np.mean(self.measurements),
            'min': min(self.measurements)
        }


class CommunicationTracker:
    """Track communication costs."""
    
    def __init__(self):
        self.bytes_sent = 0
        self.bytes_received = 0
        self.message_count = 0
        self.round_communication = []
        
    def record_send(self, data_size: int):
        """Record bytes sent."""
        self.bytes_sent += data_size
        self.message_count += 1
        
    def record_receive(self, data_size: int):
        """Record bytes received."""
        self.bytes_received += data_size
        self.message_count += 1
        
    def end_round(self) -> Dict[str, int]:
        """End current round and return statistics."""
        stats = {
            'sent': self.bytes_sent,
            'received': self.bytes_received,
            'messages': self.message_count
        }
        self.round_communication.append(stats)
        
        # Reset for next round
        self.bytes_sent = 0
        self.bytes_received = 0
        self.message_count = 0
        
        return stats
    
    def get_total(self) -> Dict[str, int]:
        """Get total communication statistics."""
        total_sent = sum(r['sent'] for r in self.round_communication)
        total_received = sum(r['received'] for r in self.round_communication)
        total_messages = sum(r['messages'] for r in self.round_communication)
        
        return {
            'total_sent': total_sent,
            'total_received': total_received,
            'total_messages': total_messages,
            'total_bytes': total_sent + total_received
        }


class MetricsCollector:
    """Main metrics collector for FL experiments."""
    
    def __init__(self, framework: str, experiment_name: str):
        """
        Initialize metrics collector.
        
        Args:
            framework: 'tff' or 'flower'
            experiment_name: Name of the experiment
        """
        self.framework = framework
        self.experiment_name = experiment_name
        self.experiment_metrics = ExperimentMetrics(
            framework=framework,
            experiment_name=experiment_name
        )
        self.memory_monitor = MemoryMonitor()
        self.communication_tracker = CommunicationTracker()
        self._round_start_time = None
        self._setup_start_time = None
        self._experiment_start_time = None
        
    def start_experiment(self, num_clients: int, num_rounds: int, **metadata):
        """Start the experiment timer."""
        self._experiment_start_time = time.time()
        self.experiment_metrics.start_time = datetime.now().isoformat()
        self.experiment_metrics.num_clients = num_clients
        self.experiment_metrics.num_rounds = num_rounds
        self.experiment_metrics.metadata = metadata
        
        # Start memory tracking with tracemalloc
        tracemalloc.start()
        self.memory_monitor.start()
        
        print(f"[{self.framework.upper()}] Experiment '{self.experiment_name}' started")
        
    def start_setup(self):
        """Start timing setup phase."""
        self._setup_start_time = time.time()
        
    def end_setup(self):
        """End setup phase timing."""
        if self._setup_start_time:
            self.experiment_metrics.setup_time_seconds = time.time() - self._setup_start_time
            
    def start_round(self, round_number: int):
        """Start timing a training round."""
        self._round_start_time = time.time()
        self._current_round = round_number
        
    def end_round(
        self,
        train_loss: float = 0.0,
        train_accuracy: float = 0.0,
        val_loss: float = 0.0,
        val_accuracy: float = 0.0,
        num_clients: int = 0,
        aggregation_time: float = 0.0
    ) -> RoundMetrics:
        """End the current round and record metrics."""
        duration = time.time() - self._round_start_time
        
        # Get communication stats
        comm_stats = self.communication_tracker.end_round()
        
        # Get current memory
        current, peak = tracemalloc.get_traced_memory()
        
        round_metrics = RoundMetrics(
            round_number=self._current_round,
            duration_seconds=duration,
            communication_bytes_sent=comm_stats['sent'],
            communication_bytes_received=comm_stats['received'],
            peak_memory_mb=peak / (1024 * 1024),
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            val_loss=val_loss,
            val_accuracy=val_accuracy,
            num_clients_participated=num_clients,
            aggregation_time_seconds=aggregation_time
        )
        
        self.experiment_metrics.rounds.append(round_metrics)
        
        print(f"[{self.framework.upper()}] Round {self._current_round}: "
              f"loss={train_loss:.4f}, acc={train_accuracy:.4f}, "
              f"time={duration:.2f}s")
        
        return round_metrics
    
    def record_model_size(self, size_bytes: int):
        """Record the model size in bytes."""
        self.experiment_metrics.model_size_bytes = size_bytes
        
    def record_communication(self, sent: int = 0, received: int = 0):
        """Record communication data."""
        if sent:
            self.communication_tracker.record_send(sent)
        if received:
            self.communication_tracker.record_receive(received)
            
    def record_error(self, error: str):
        """Record an error message."""
        self.experiment_metrics.errors.append(error)
        
    def end_experiment(
        self,
        final_accuracy: float = 0.0,
        final_loss: float = 0.0
    ) -> ExperimentMetrics:
        """End the experiment and finalize metrics."""
        self.experiment_metrics.end_time = datetime.now().isoformat()
        self.experiment_metrics.total_duration_seconds = (
            time.time() - self._experiment_start_time
        )
        
        # Stop memory monitoring
        mem_stats = self.memory_monitor.stop()
        tracemalloc.stop()
        
        self.experiment_metrics.peak_memory_mb = mem_stats['peak']
        self.experiment_metrics.final_accuracy = final_accuracy
        self.experiment_metrics.final_loss = final_loss
        
        # Calculate total communication
        comm_total = self.communication_tracker.get_total()
        self.experiment_metrics.total_communication_bytes = comm_total['total_bytes']
        
        print(f"\n[{self.framework.upper()}] Experiment completed:")
        print(f"  Duration: {self.experiment_metrics.total_duration_seconds:.2f}s")
        print(f"  Peak Memory: {self.experiment_metrics.peak_memory_mb:.2f} MB")
        print(f"  Communication: {comm_total['total_bytes'] / 1024:.2f} KB")
        print(f"  Final Accuracy: {final_accuracy:.4f}")
        
        return self.experiment_metrics
    
    def save_results(self, filepath: str):
        """Save experiment metrics to JSON file."""
        # Convert to dictionary
        metrics_dict = asdict(self.experiment_metrics)
        
        # Convert RoundMetrics to dicts
        metrics_dict['rounds'] = [asdict(r) for r in self.experiment_metrics.rounds]
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
            
        print(f"Results saved to {filepath}")
        
    @staticmethod
    def load_results(filepath: str) -> Dict:
        """Load experiment metrics from JSON file."""
        with open(filepath, 'r') as f:
            return json.load(f)


class FlexibilityMetrics:
    """Collect flexibility-related metrics."""
    
    @staticmethod
    def count_lines_of_code(filepath: str) -> int:
        """Count non-empty, non-comment lines of code."""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        count = 0
        in_multiline_string = False
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                continue
                
            # Skip single-line comments
            if stripped.startswith('#'):
                continue
                
            # Handle multiline strings (docstrings)
            if '"""' in stripped or "'''" in stripped:
                count_quotes = stripped.count('"""') + stripped.count("'''")
                if count_quotes == 1:
                    in_multiline_string = not in_multiline_string
                continue
                
            if in_multiline_string:
                continue
                
            count += 1
            
        return count
    
    @staticmethod
    def count_dependencies(requirements_file: str) -> int:
        """Count number of dependencies in requirements file."""
        with open(requirements_file, 'r') as f:
            lines = f.readlines()
        
        count = 0
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                count += 1
                
        return count
    
    @staticmethod
    def measure_import_time(module_name: str) -> float:
        """Measure time to import a module."""
        import importlib
        
        # Clear module from cache if present
        if module_name in sys.modules:
            del sys.modules[module_name]
        
        start = time.time()
        try:
            importlib.import_module(module_name)
        except ImportError:
            return -1
        
        return time.time() - start


def compare_experiments(
    tff_results: Dict,
    flower_results: Dict
) -> Dict[str, Dict]:
    """Compare metrics between TFF and Flower experiments."""
    
    comparison = {
        'performance': {
            'tff_final_accuracy': tff_results['final_accuracy'],
            'flower_final_accuracy': flower_results['final_accuracy'],
            'tff_final_loss': tff_results['final_loss'],
            'flower_final_loss': flower_results['final_loss'],
        },
        'efficiency': {
            'tff_total_time': tff_results['total_duration_seconds'],
            'flower_total_time': flower_results['total_duration_seconds'],
            'tff_setup_time': tff_results['setup_time_seconds'],
            'flower_setup_time': flower_results['setup_time_seconds'],
        },
        'memory': {
            'tff_peak_memory_mb': tff_results['peak_memory_mb'],
            'flower_peak_memory_mb': flower_results['peak_memory_mb'],
        },
        'communication': {
            'tff_total_bytes': tff_results['total_communication_bytes'],
            'flower_total_bytes': flower_results['total_communication_bytes'],
        }
    }
    
    # Calculate relative differences
    comparison['relative_differences'] = {
        'accuracy_diff': (
            flower_results['final_accuracy'] - tff_results['final_accuracy']
        ),
        'time_ratio': (
            tff_results['total_duration_seconds'] / 
            max(flower_results['total_duration_seconds'], 0.001)
        ),
        'memory_ratio': (
            tff_results['peak_memory_mb'] / 
            max(flower_results['peak_memory_mb'], 0.001)
        ),
        'communication_ratio': (
            tff_results['total_communication_bytes'] / 
            max(flower_results['total_communication_bytes'], 1)
        )
    }
    
    return comparison


if __name__ == "__main__":
    # Example usage
    collector = MetricsCollector('test_framework', 'example_experiment')
    
    collector.start_experiment(num_clients=5, num_rounds=3)
    collector.start_setup()
    time.sleep(0.1)  # Simulate setup
    collector.end_setup()
    
    for round_num in range(3):
        collector.start_round(round_num + 1)
        time.sleep(0.1)  # Simulate training
        collector.record_communication(sent=1000, received=500)
        collector.end_round(
            train_loss=0.5 - round_num * 0.1,
            train_accuracy=0.7 + round_num * 0.05,
            num_clients=5
        )
    
    metrics = collector.end_experiment(final_accuracy=0.85, final_loss=0.2)
    print(f"\nExperiment metrics: {metrics}")
