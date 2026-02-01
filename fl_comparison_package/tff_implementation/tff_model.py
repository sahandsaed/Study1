"""
TensorFlow Model for Bug Prediction
Used in TensorFlow Federated implementation
"""

import tensorflow as tf
from typing import Tuple, Optional


def create_bug_prediction_model(
    vocab_size: int,
    embedding_dim: int = 128,
    lstm_units: int = 64,
    num_features: int = 20,
    dropout_rate: float = 0.3
) -> tf.keras.Model:
    """
    Create a hybrid model combining code embeddings and handcrafted features.
    
    Architecture:
    - Token input → Embedding → BiLSTM → Dense
    - Feature input → Dense
    - Concatenate → Dense → Output
    
    Args:
        vocab_size: Size of token vocabulary
        embedding_dim: Dimension of token embeddings
        lstm_units: Number of LSTM units
        num_features: Number of handcrafted features
        dropout_rate: Dropout rate
        
    Returns:
        Keras model
    """
    # Token input branch
    token_input = tf.keras.layers.Input(shape=(None,), name='token_input')
    
    # Embedding layer
    x = tf.keras.layers.Embedding(
        vocab_size, 
        embedding_dim,
        mask_zero=True,
        name='embedding'
    )(token_input)
    
    # Bidirectional LSTM
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units, return_sequences=True),
        name='bilstm_1'
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(lstm_units // 2),
        name='bilstm_2'
    )(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Feature input branch
    feature_input = tf.keras.layers.Input(shape=(num_features,), name='feature_input')
    y = tf.keras.layers.Dense(32, activation='relu', name='feature_dense')(feature_input)
    y = tf.keras.layers.Dropout(dropout_rate)(y)
    
    # Concatenate branches
    combined = tf.keras.layers.Concatenate(name='concat')([x, y])
    
    # Final classification layers
    z = tf.keras.layers.Dense(64, activation='relu', name='final_dense_1')(combined)
    z = tf.keras.layers.Dropout(dropout_rate)(z)
    z = tf.keras.layers.Dense(32, activation='relu', name='final_dense_2')(z)
    
    # Output layer
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(z)
    
    model = tf.keras.Model(
        inputs=[token_input, feature_input],
        outputs=output,
        name='bug_prediction_model'
    )
    
    return model


def create_simple_model(
    vocab_size: int,
    embedding_dim: int = 64,
    max_length: int = 256
) -> tf.keras.Model:
    """
    Create a simpler model using only token embeddings.
    Better for resource-constrained FL scenarios.
    
    Args:
        vocab_size: Size of token vocabulary
        embedding_dim: Dimension of token embeddings
        max_length: Maximum sequence length
        
    Returns:
        Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(max_length,)),
        tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ], name='simple_bug_predictor')
    
    return model


def create_feature_only_model(num_features: int = 20) -> tf.keras.Model:
    """
    Create a model using only handcrafted features.
    Useful as a baseline.
    
    Args:
        num_features: Number of input features
        
    Returns:
        Keras model
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(num_features,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ], name='feature_only_predictor')
    
    return model


def get_model_size(model: tf.keras.Model) -> int:
    """
    Calculate model size in bytes.
    
    Args:
        model: Keras model
        
    Returns:
        Size in bytes
    """
    import numpy as np
    
    total_size = 0
    for weight in model.get_weights():
        total_size += weight.nbytes
    
    return total_size


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 0.001
) -> tf.keras.Model:
    """
    Compile model with standard settings for bug prediction.
    
    Args:
        model: Keras model
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled model
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


class ModelBuilder:
    """Builder class for creating TFF-compatible models."""
    
    def __init__(
        self,
        vocab_size: int,
        max_length: int = 256,
        num_features: int = 20,
        model_type: str = 'simple'
    ):
        """
        Initialize model builder.
        
        Args:
            vocab_size: Vocabulary size for embeddings
            max_length: Maximum sequence length
            num_features: Number of handcrafted features
            model_type: 'simple', 'hybrid', or 'feature_only'
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_features = num_features
        self.model_type = model_type
        
    def build(self) -> tf.keras.Model:
        """Build and return the model."""
        if self.model_type == 'simple':
            model = create_simple_model(
                self.vocab_size,
                max_length=self.max_length
            )
        elif self.model_type == 'hybrid':
            model = create_bug_prediction_model(
                self.vocab_size,
                num_features=self.num_features
            )
        elif self.model_type == 'feature_only':
            model = create_feature_only_model(self.num_features)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        return compile_model(model)
    
    def get_input_spec(self):
        """Get TFF input specification."""
        if self.model_type == 'simple':
            return (
                tf.TensorSpec(shape=[None, self.max_length], dtype=tf.int32),
                tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
            )
        elif self.model_type == 'hybrid':
            return (
                (
                    tf.TensorSpec(shape=[None, self.max_length], dtype=tf.int32),
                    tf.TensorSpec(shape=[None, self.num_features], dtype=tf.float32)
                ),
                tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
            )
        else:  # feature_only
            return (
                tf.TensorSpec(shape=[None, self.num_features], dtype=tf.float32),
                tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
            )


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Simple model
    simple_model = create_simple_model(vocab_size=5000)
    simple_model.summary()
    print(f"Simple model size: {get_model_size(simple_model) / 1024:.2f} KB\n")
    
    # Hybrid model
    hybrid_model = create_bug_prediction_model(vocab_size=5000)
    hybrid_model.summary()
    print(f"Hybrid model size: {get_model_size(hybrid_model) / 1024:.2f} KB\n")
    
    # Feature-only model
    feature_model = create_feature_only_model()
    feature_model.summary()
    print(f"Feature model size: {get_model_size(feature_model) / 1024:.2f} KB")
