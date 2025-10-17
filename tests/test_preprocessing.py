"""Unit tests for text preprocessing functions"""
import unittest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import SentimentAnalyzer


class TestPreprocessing(unittest.TestCase):
    """Test cases for text preprocessing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = SentimentAnalyzer()
    
    def test_lowercase_conversion(self):
        """Test that text is converted to lowercase"""
        text = "THIS IS UPPERCASE TEXT"
        result = self.analyzer.preprocess_text(text)
        self.assertEqual(result, result.lower())
    
    def test_url_removal(self):
        """Test that URLs are removed from text"""
        text = "Check out https://example.com for more info"
        result = self.analyzer.preprocess_text(text)
        self.assertNotIn('http', result)
        self.assertNotIn('www', result)
    
    def test_special_characters_removal(self):
        """Test that special characters are removed"""
        text = "Hello! This is a test... #testing @mentions"
        result = self.analyzer.preprocess_text(text)
        # Should only contain letters and spaces
        self.assertTrue(all(c.isalpha() or c.isspace() for c in result))
    
    def test_stopwords_removal(self):
        """Test that stopwords are removed"""
        text = "this is a test with some stopwords"
        result = self.analyzer.preprocess_text(text)
        # Common stopwords should be removed
        self.assertNotIn('this', result.split())
        self.assertNotIn('is', result.split())
        self.assertNotIn('a', result.split())
    
    def test_empty_text(self):
        """Test handling of empty text"""
        text = ""
        result = self.analyzer.preprocess_text(text)
        self.assertEqual(result, "")
    
    def test_text_with_only_stopwords(self):
        """Test text containing only stopwords"""
        text = "the is a an"
        result = self.analyzer.preprocess_text(text)
        # Result should be empty or very short
        self.assertTrue(len(result) < len(text))


class TestSentimentPrediction(unittest.TestCase):
    """Test cases for sentiment prediction"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = SentimentAnalyzer()
    
    def test_positive_sentiment(self):
        """Test detection of positive sentiment"""
        text = "This is amazing and wonderful! I love it!"
        result = self.analyzer.predict(text)
        self.assertEqual(result['sentiment'], 'positive')
    
    def test_negative_sentiment(self):
        """Test detection of negative sentiment"""
        text = "This is terrible and awful. I hate it!"
        result = self.analyzer.predict(text)
        self.assertEqual(result['sentiment'], 'negative')
    
    def test_neutral_sentiment(self):
        """Test detection of neutral sentiment"""
        text = "This is a product."
        result = self.analyzer.predict(text)
        self.assertEqual(result['sentiment'], 'neutral')
    
    def test_multiple_texts(self):
        """Test prediction on multiple texts"""
        texts = [
            "Great product!",
            "Terrible service.",
            "It's okay."
        ]
        results = self.analyzer.predict(texts)
        self.assertEqual(len(results), 3)
        self.assertTrue(all('sentiment' in r for r in results))
        self.assertTrue(all('confidence' in r for r in results))
    
    def test_confidence_score_range(self):
        """Test that confidence scores are between 0 and 1"""
        text = "This is a test"
        result = self.analyzer.predict(text)
        self.assertGreaterEqual(result['confidence'], 0.0)
        self.assertLessEqual(result['confidence'], 1.0)


if __name__ == '__main__':
    unittest.main()
