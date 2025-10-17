#!/usr/bin/env python3
"""
Batch Sentiment Analysis Script

This script performs sentiment analysis on multiple texts from a CSV file.
It reads input from a CSV, predicts sentiment for each text, and saves results to a new CSV.

Usage:
    python batch_predict.py --input data.csv --output results.csv --text-column text
"""

import argparse
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# Default paths
DEFAULT_MODEL_PATH = "models/sentiment_model.pkl"
DEFAULT_VECTORIZER_PATH = "models/vectorizer.pkl"

class BatchSentimentPredictor:
    """Batch sentiment prediction class"""
    
    def __init__(self, model_path=DEFAULT_MODEL_PATH, vectorizer_path=DEFAULT_VECTORIZER_PATH):
        """
        Initialize the batch predictor
        
        Args:
            model_path: Path to the trained model pickle file
            vectorizer_path: Path to the vectorizer pickle file
        """
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.model = None
        self.vectorizer = None
        self.sentiment_labels = ['negative', 'positive']  # Adjust based on your model
        
    def load_model(self):
        """Load the trained model and vectorizer"""
        try:
            print(f"Loading model from {self.model_path}...")
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully!")
            
            print(f"Loading vectorizer from {self.vectorizer_path}...")
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("Vectorizer loaded successfully!")
            
            return True
        except FileNotFoundError as e:
            print(f"Error: Could not find model files. {e}")
            print("Please ensure model files exist in the 'models/' directory.")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict_single(self, text):
        """
        Predict sentiment for a single text
        
        Args:
            text: Input text string
            
        Returns:
            dict: Dictionary with sentiment, confidence, and probabilities
        """
        if not text or pd.isna(text):
            return {
                'sentiment': 'unknown',
                'confidence': 0.0,
                'negative_prob': 0.0,
                'positive_prob': 0.0
            }
        
        # Vectorize text
        text_vectorized = self.vectorizer.transform([str(text)])
        
        # Predict
        prediction = self.model.predict(text_vectorized)[0]
        probabilities = self.model.predict_proba(text_vectorized)[0]
        
        sentiment = self.sentiment_labels[prediction]
        confidence = float(probabilities[prediction])
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'negative_prob': float(probabilities[0]),
            'positive_prob': float(probabilities[1])
        }
    
    def predict_batch(self, texts):
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of text strings
            
        Returns:
            list: List of prediction dictionaries
        """
        results = []
        total = len(texts)
        
        print(f"\nProcessing {total} texts...")
        for i, text in enumerate(texts, 1):
            if i % 100 == 0:
                print(f"Progress: {i}/{total} ({i/total*100:.1f}%)")
            
            result = self.predict_single(text)
            results.append(result)
        
        print(f"Completed: {total}/{total} (100%)")
        return results
    
    def process_csv(self, input_path, output_path, text_column='text', id_column=None):
        """
        Process a CSV file and save results
        
        Args:
            input_path: Path to input CSV file
            output_path: Path to output CSV file
            text_column: Name of the column containing text to analyze
            id_column: Optional name of ID column to preserve
        """
        try:
            # Read input CSV
            print(f"\nReading input file: {input_path}")
            df = pd.read_csv(input_path)
            print(f"Loaded {len(df)} rows")
            
            # Validate text column
            if text_column not in df.columns:
                print(f"Error: Column '{text_column}' not found in CSV.")
                print(f"Available columns: {', '.join(df.columns)}")
                return False
            
            # Perform batch prediction
            predictions = self.predict_batch(df[text_column].tolist())
            
            # Create results dataframe
            results_df = pd.DataFrame(predictions)
            
            # Combine with original data
            if id_column and id_column in df.columns:
                results_df.insert(0, id_column, df[id_column])
            
            results_df.insert(0, text_column, df[text_column])
            
            # Add timestamp
            results_df['predicted_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Save results
            print(f"\nSaving results to: {output_path}")
            results_df.to_csv(output_path, index=False)
            print(f"Results saved successfully!")
            
            # Print summary statistics
            self.print_summary(results_df)
            
            return True
            
        except Exception as e:
            print(f"Error processing CSV: {e}")
            return False
    
    def print_summary(self, df):
        """Print summary statistics of predictions"""
        print("\n" + "="*50)
        print("PREDICTION SUMMARY")
        print("="*50)
        
        # Count sentiments
        sentiment_counts = df['sentiment'].value_counts()
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")
        
        # Average confidence
        avg_confidence = df['confidence'].mean()
        print(f"\nAverage Confidence: {avg_confidence:.3f}")
        
        # Confidence by sentiment
        print("\nAverage Confidence by Sentiment:")
        for sentiment in df['sentiment'].unique():
            if sentiment != 'unknown':
                avg_conf = df[df['sentiment'] == sentiment]['confidence'].mean()
                print(f"  {sentiment.capitalize()}: {avg_conf:.3f}")
        
        print("\n" + "="*50)

def main():
    """Main function to run batch prediction"""
    parser = argparse.ArgumentParser(
        description='Batch Sentiment Analysis - Predict sentiment for multiple texts from CSV'
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input CSV file path'
    )
    
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output CSV file path for results'
    )
    
    parser.add_argument(
        '--text-column', '-t',
        default='text',
        help='Name of the column containing text to analyze (default: text)'
    )
    
    parser.add_argument(
        '--id-column', '-id',
        default=None,
        help='Name of the ID column to preserve (optional)'
    )
    
    parser.add_argument(
        '--model-path', '-m',
        default=DEFAULT_MODEL_PATH,
        help=f'Path to model file (default: {DEFAULT_MODEL_PATH})'
    )
    
    parser.add_argument(
        '--vectorizer-path', '-v',
        default=DEFAULT_VECTORIZER_PATH,
        help=f'Path to vectorizer file (default: {DEFAULT_VECTORIZER_PATH})'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("="*50)
    print("BATCH SENTIMENT ANALYSIS")
    print("="*50)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Text column: {args.text_column}")
    if args.id_column:
        print(f"ID column: {args.id_column}")
    print("="*50)
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"\nError: Input file '{args.input}' not found.")
        sys.exit(1)
    
    # Initialize predictor
    predictor = BatchSentimentPredictor(
        model_path=args.model_path,
        vectorizer_path=args.vectorizer_path
    )
    
    # Load model
    if not predictor.load_model():
        sys.exit(1)
    
    # Process CSV
    success = predictor.process_csv(
        input_path=args.input,
        output_path=args.output,
        text_column=args.text_column,
        id_column=args.id_column
    )
    
    if success:
        print("\n✓ Batch prediction completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Batch prediction failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
