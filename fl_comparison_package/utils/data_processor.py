"""
Data Processor for Bug Prediction Dataset
Handles loading, preprocessing, and partitioning for FL experiments
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.model_selection import train_test_split
import hashlib


class BugPredictionDataProcessor:
    """Process bug prediction dataset for federated learning."""
    
    def __init__(self, data_path: str):
        """
        Initialize the data processor.
        
        Args:
            data_path: Path to the JSON dataset file
        """
        self.data_path = data_path
        self.data = None
        self.processed_data = None
        
    def load_data(self) -> List[Dict]:
        """Load the dataset from JSON file."""
        with open(self.data_path, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} code pairs")
        return self.data
    
    def get_exception_types(self) -> Dict[str, int]:
        """Get distribution of exception types in dataset."""
        if self.data is None:
            self.load_data()
        
        exception_counts = {}
        for item in self.data:
            exc_type = item.get('exception_type', 'Unknown')
            exception_counts[exc_type] = exception_counts.get(exc_type, 0) + 1
        
        return exception_counts
    
    def prepare_binary_classification(self) -> Tuple[List[str], List[int]]:
        """
        Prepare data for binary bug classification.
        
        Returns:
            Tuple of (code_samples, labels) where:
            - code_samples: List of code strings
            - labels: 0 for buggy, 1 for fixed
        """
        if self.data is None:
            self.load_data()
        
        code_samples = []
        labels = []
        
        for item in self.data:
            # Add buggy code with label 0
            code_samples.append(item['buggy'])
            labels.append(0)
            
            # Add fixed code with label 1
            code_samples.append(item['fixed'])
            labels.append(1)
        
        self.processed_data = (code_samples, labels)
        print(f"Prepared {len(code_samples)} samples for binary classification")
        print(f"  - Buggy samples: {labels.count(0)}")
        print(f"  - Fixed samples: {labels.count(1)}")
        
        return code_samples, labels
    
    def partition_iid(
        self, 
        num_clients: int,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Dict]:
        """
        Partition data IID (Independent and Identically Distributed).
        
        Args:
            num_clients: Number of FL clients
            test_size: Fraction for test set
            random_state: Random seed
            
        Returns:
            Dictionary with client data and global test set
        """
        if self.processed_data is None:
            self.prepare_binary_classification()
        
        code_samples, labels = self.processed_data
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            code_samples, labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=labels
        )
        
        # Shuffle training data
        indices = np.random.RandomState(random_state).permutation(len(X_train))
        X_train = [X_train[i] for i in indices]
        y_train = [y_train[i] for i in indices]
        
        # Partition among clients
        partition_size = len(X_train) // num_clients
        client_data = {}
        
        for client_id in range(num_clients):
            start_idx = client_id * partition_size
            end_idx = start_idx + partition_size if client_id < num_clients - 1 else len(X_train)
            
            client_data[f"client_{client_id}"] = {
                "X": X_train[start_idx:end_idx],
                "y": y_train[start_idx:end_idx]
            }
        
        return {
            "clients": client_data,
            "test": {"X": X_test, "y": y_test},
            "partition_type": "iid"
        }
    
    def partition_non_iid_by_exception(
        self, 
        num_clients: int,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Dict]:
        """
        Partition data Non-IID by exception type.
        Each client gets samples primarily from specific exception types.
        
        Args:
            num_clients: Number of FL clients
            test_size: Fraction for test set
            random_state: Random seed
            
        Returns:
            Dictionary with client data and global test set
        """
        if self.data is None:
            self.load_data()
        
        # Group by exception type
        exception_groups = {}
        for item in self.data:
            exc_type = item.get('exception_type', 'Unknown')
            if exc_type not in exception_groups:
                exception_groups[exc_type] = []
            exception_groups[exc_type].append(item)
        
        # Assign exception types to clients
        exception_types = list(exception_groups.keys())
        np.random.RandomState(random_state).shuffle(exception_types)
        
        # Distribute exception types among clients
        client_exceptions = {f"client_{i}": [] for i in range(num_clients)}
        for i, exc_type in enumerate(exception_types):
            client_id = i % num_clients
            client_exceptions[f"client_{client_id}"].append(exc_type)
        
        # Create client datasets
        client_data = {}
        all_test_X = []
        all_test_y = []
        
        for client_id, exc_types in client_exceptions.items():
            client_X = []
            client_y = []
            
            for exc_type in exc_types:
                items = exception_groups[exc_type]
                
                # Split this exception type's data
                train_items, test_items = train_test_split(
                    items, test_size=test_size, random_state=random_state
                )
                
                # Add training data
                for item in train_items:
                    client_X.extend([item['buggy'], item['fixed']])
                    client_y.extend([0, 1])
                
                # Add test data to global test set
                for item in test_items:
                    all_test_X.extend([item['buggy'], item['fixed']])
                    all_test_y.extend([0, 1])
            
            client_data[client_id] = {"X": client_X, "y": client_y}
        
        return {
            "clients": client_data,
            "test": {"X": all_test_X, "y": all_test_y},
            "partition_type": "non_iid_exception",
            "client_exceptions": client_exceptions
        }
    
    def get_data_statistics(self, partitioned_data: Dict) -> Dict:
        """Get statistics about partitioned data."""
        stats = {
            "partition_type": partitioned_data["partition_type"],
            "num_clients": len(partitioned_data["clients"]),
            "test_size": len(partitioned_data["test"]["X"]),
            "client_stats": {}
        }
        
        for client_id, data in partitioned_data["clients"].items():
            stats["client_stats"][client_id] = {
                "total_samples": len(data["X"]),
                "buggy_samples": data["y"].count(0) if isinstance(data["y"], list) else np.sum(np.array(data["y"]) == 0),
                "fixed_samples": data["y"].count(1) if isinstance(data["y"], list) else np.sum(np.array(data["y"]) == 1)
            }
        
        return stats


def get_sample_hash(code: str) -> str:
    """Generate a hash for a code sample (for deduplication)."""
    return hashlib.md5(code.encode()).hexdigest()


if __name__ == "__main__":
    # Example usage
    processor = BugPredictionDataProcessor("data/dataset_pairs_1_.json")
    processor.load_data()
    
    # Show exception distribution
    print("\nException Type Distribution:")
    for exc_type, count in processor.get_exception_types().items():
        print(f"  {exc_type}: {count}")
    
    # Prepare and partition data
    processor.prepare_binary_classification()
    
    # IID partition
    iid_data = processor.partition_iid(num_clients=5)
    print("\nIID Partition Statistics:")
    stats = processor.get_data_statistics(iid_data)
    for client_id, client_stats in stats["client_stats"].items():
        print(f"  {client_id}: {client_stats['total_samples']} samples")
    
    # Non-IID partition
    non_iid_data = processor.partition_non_iid_by_exception(num_clients=5)
    print("\nNon-IID Partition Statistics:")
    stats = processor.get_data_statistics(non_iid_data)
    for client_id, client_stats in stats["client_stats"].items():
        print(f"  {client_id}: {client_stats['total_samples']} samples")
