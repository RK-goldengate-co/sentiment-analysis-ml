"""Model definitions for sentiment analysis.

This module contains the implementation of machine learning and deep learning models
for sentiment classification.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
import os


class SentimentModel:
    """Base class for sentiment analysis models."""
    
    def __init__(self, model_type='lstm'):
        """
        Initialize the sentiment model.
        
        Args:
            model_type (str): Type of model ('lstm', 'cnn', 'transformer', 'sklearn')
        """
        self.model_type = model_type
        self.model = None
        self.vectorizer = None
        self.max_length = 200
        self.embedding_dim = 128
        self.num_classes = 3  # Positive, Negative, Neutral
        
    def build_lstm_model(self, vocab_size):
        """Build LSTM-based sentiment model."""
        model = keras.Sequential([
            layers.Embedding(vocab_size, self.embedding_dim, input_length=self.max_length),
            layers.SpatialDropout1D(0.2),
            layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def build_cnn_model(self, vocab_size):
        """Build CNN-based sentiment model."""
        model = keras.Sequential([
            layers.Embedding(vocab_size, self.embedding_dim, input_length=self.max_length),
            layers.Conv1D(128, 5, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def build_transformer_model(self, vocab_size):
        """Build Transformer-based sentiment model."""
        # Input layer
        inputs = layers.Input(shape=(self.max_length,))
        
        # Embedding layer
        x = layers.Embedding(vocab_size, self.embedding_dim)(inputs)
        
        # Transformer block
        attention_output = layers.MultiHeadAttention(
            num_heads=8, 
            key_dim=self.embedding_dim
        )(x, x)
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)
        
        # Feed forward
        ff_output = layers.Dense(256, activation='relu')(x)
        ff_output = layers.Dense(self.embedding_dim)(ff_output)
        x = layers.Add()([x, ff_output])
        x = layers.LayerNormalization()(x)
        
        # Global average pooling and classification
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model
    
    def build_sklearn_model(self, algorithm='logistic'):
        """Build scikit-learn based sentiment model."""
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        if algorithm == 'logistic':
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif algorithm == 'naive_bayes':
            model = MultinomialNB()
        elif algorithm == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        return model
    
    def build_model(self, vocab_size=10000, algorithm='logistic'):
        """Build the model based on model_type."""
        if self.model_type == 'lstm':
            self.model = self.build_lstm_model(vocab_size)
        elif self.model_type == 'cnn':
            self.model = self.build_cnn_model(vocab_size)
        elif self.model_type == 'transformer':
            self.model = self.build_transformer_model(vocab_size)
        elif self.model_type == 'sklearn':
            self.model = self.build_sklearn_model(algorithm)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def save_model(self, filepath):
        """Save the trained model."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.model_type == 'sklearn':
            with open(filepath, 'wb') as f:
                pickle.dump({'model': self.model, 'vectorizer': self.vectorizer}, f)
        else:
            self.model.save(filepath)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        if self.model_type == 'sklearn':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.vectorizer = data['vectorizer']
        else:
            self.model = keras.models.load_model(filepath)
        
        print(f"Model loaded from {filepath}")
        return self.model
    
    def get_model_summary(self):
        """Get model summary."""
        if self.model_type == 'sklearn':
            return f"Sklearn model: {type(self.model).__name__}"
        else:
            return self.model.summary()


class EnsembleModel:
    """Ensemble model combining multiple sentiment models."""
    
    def __init__(self, models):
        """
        Initialize ensemble model.
        
        Args:
            models (list): List of trained models
        """
        self.models = models
    
    def predict(self, X, weights=None):
        """Make predictions using ensemble."""
        if weights is None:
            weights = [1.0 / len(self.models)] * len(self.models)
        
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        return ensemble_pred


if __name__ == "__main__":
    # Example usage
    print("Sentiment Analysis Models")
    print("=" * 50)
    
    # Build different models
    model_types = ['lstm', 'cnn', 'transformer', 'sklearn']
    
    for model_type in model_types:
        print(f"\nBuilding {model_type.upper()} model...")
        sentiment_model = SentimentModel(model_type=model_type)
        
        if model_type == 'sklearn':
            model = sentiment_model.build_model(algorithm='logistic')
        else:
            model = sentiment_model.build_model(vocab_size=10000)
        
        print(f"{model_type.upper()} model built successfully!")
        print(sentiment_model.get_model_summary())
