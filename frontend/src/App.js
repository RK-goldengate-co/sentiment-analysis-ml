import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const analyzeSentiment = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${API_URL}/predict`, {
        text: text
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Error analyzing sentiment. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (sentiment) => {
    switch(sentiment?.toLowerCase()) {
      case 'positive':
        return '#4CAF50';
      case 'negative':
        return '#f44336';
      case 'neutral':
        return '#FF9800';
      default:
        return '#2196F3';
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🎭 Sentiment Analysis</h1>
        <p>Analyze the sentiment of any text using Machine Learning</p>
      </header>

      <div className="container">
        <div className="input-section">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to analyze sentiment..."
            rows="6"
            className="text-input"
          />
          <button 
            onClick={analyzeSentiment} 
            disabled={loading}
            className="analyze-btn"
          >
            {loading ? 'Analyzing...' : 'Analyze Sentiment'}
          </button>
        </div>

        {error && (
          <div className="error-message">
            ⚠️ {error}
          </div>
        )}

        {result && (
          <div className="result-section">
            <h2>Analysis Result</h2>
            <div className="result-card">
              <div className="sentiment" style={{ color: getSentimentColor(result.sentiment) }}>
                <span className="label">Sentiment:</span>
                <span className="value">{result.sentiment}</span>
              </div>
              <div className="confidence">
                <span className="label">Confidence:</span>
                <span className="value">{(result.confidence * 100).toFixed(2)}%</span>
              </div>
              <div className="confidence-bar">
                <div 
                  className="confidence-fill" 
                  style={{ 
                    width: `${result.confidence * 100}%`,
                    backgroundColor: getSentimentColor(result.sentiment)
                  }}
                />
              </div>
            </div>
          </div>
        )}
      </div>

      <footer className="App-footer">
        <p>Powered by FastAPI & React | ML Sentiment Analysis</p>
      </footer>
    </div>
  );
}

export default App;
