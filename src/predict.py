"""Sentiment Analysis Prediction Module"""
import numpy as np
import pickle
from typing import Dict, Union
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class SentimentAnalyzer:
    """Simple Sentiment Analyzer for text classification"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize the sentiment analyzer
        
        Args:
            model_path: Path to pre-trained model (optional)
        """
        self.model = None
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        self.stop_words = set(stopwords.words('english'))
        
        if model_path:
            self.load_model(model_path)
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis
        
        Args:
            text: Input text string
            
        Returns:
            Cleaned and preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens 
                  if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(tokens)
    
    def predict(self, text: Union[str, list]) -> Dict:
        """
        Predict sentiment for given text
        
        Args:
            text: Input text or list of texts
            
        Returns:
            Dictionary with sentiment and confidence score
        """
        if isinstance(text, str):
            text = [text]
        
        # Preprocess
        processed_texts = [self.preprocess_text(t) for t in text]
        
        # Simple rule-based sentiment (placeholder for ML model)
        results = []
        for proc_text in processed_texts:
            sentiment, confidence = self._rule_based_sentiment(proc_text)
            results.append({
                'text': text[len(results)],
                'sentiment': sentiment,
                'confidence': confidence
            })
        
        return results[0] if len(results) == 1 else results
    
    def _rule_based_sentiment(self, text: str) -> tuple:
        """
        Simple rule-based sentiment analysis (placeholder)
        
        Args:
            text: Preprocessed text
            
        Returns:
            Tuple of (sentiment, confidence)
        """
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful',
                         'fantastic', 'love', 'best', 'perfect', 'awesome']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst',
                         'hate', 'poor', 'disappointing', 'useless', 'waste']
        
        words = text.split()
        pos_count = sum(1 for word in words if word in positive_words)
        neg_count = sum(1 for word in words if word in negative_words)
        
        total = pos_count + neg_count
        
        if total == 0:
            return 'neutral', 0.5
        
        if pos_count > neg_count:
            confidence = pos_count / total
            return 'positive', confidence
        elif neg_count > pos_count:
            confidence = neg_count / total
            return 'negative', confidence
        else:
            return 'neutral', 0.5
    
    def load_model(self, model_path: str):
        """Load pre-trained model"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def save_model(self, model_path: str):
        """Save trained model"""
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")


if __name__ == "__main__":
    # Example usage
    analyzer = SentimentAnalyzer()
    
    # Test sentences
    test_texts = [
        "This product is amazing! I love it.",
        "Terrible experience. Would not recommend.",
        "It's okay, nothing special."
    ]
    
    print("Sentiment Analysis Results:")
    print("=" * 50)
    
    for text in test_texts:
        result = analyzer.predict(text)
        print(f"\nText: {result['text']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.2f}")
