# Data Directory

This directory contains all data files used in the sentiment analysis project.

## Structure

### raw/
Contains raw, unprocessed data files:
- Original datasets from various sources
- Collected tweets, reviews, feedback, etc.
- Data should be in CSV, JSON, or TXT format

### processed/
Contains preprocessed data ready for training:
- Cleaned and tokenized text
- Feature-extracted datasets
- Train/validation/test splits

## Data Sources

Recommended public datasets for sentiment analysis:

1. **IMDB Movie Reviews**: 50,000 movie reviews labeled as positive/negative
2. **Twitter Sentiment140**: 1.6 million tweets with sentiment labels
3. **Amazon Product Reviews**: Customer reviews with ratings
4. **Yelp Reviews**: Restaurant and business reviews
5. **Stanford Sentiment Treebank**: Fine-grained sentiment labels

## Data Format

Expected CSV format for training:

```csv
text,sentiment
"This is a great product!",positive
"I hate this service.",negative
"It's okay, nothing special.",neutral
```

## Usage

```python
import pandas as pd

# Load data
train_data = pd.read_csv('data/processed/train.csv')
test_data = pd.read_csv('data/processed/test.csv')
```

## Data Privacy

- Do not commit sensitive or personal data
- Always anonymize data before committing
- Add `data/raw/*.csv` and `data/processed/*.csv` to .gitignore if necessary
