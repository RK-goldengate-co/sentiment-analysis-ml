# Sentiment Analysis ML Project

## 📋 Mô tả dự án

Dự án phân tích cảm xúc (Sentiment Analysis) sử dụng Machine Learning và Deep Learning để phân loại cảm xúc trong văn bản. Hệ thống có khả năng phân tích và xác định cảm xúc tích cực, tiêu cực hoặc trung lập từ các đoạn văn bản tiếng Anh.

## 🎯 Mục tiêu

- Xây dựng mô hình phân loại cảm xúc chính xác cao
- Hỗ trợ phân tích dữ liệu văn bản từ nhiều nguồn (reviews, social media, feedback)
- Cung cấp API đơn giản để tích hợp vào các ứng dụng khác
- Nghiên cứu và so sánh hiệu suất của các thuật toán ML/DL khác nhau

## 🔧 Công nghệ sử dụng

- **Python 3.8+**: Ngôn ngữ lập trình chính
- **NumPy & Pandas**: Xử lý và phân tích dữ liệu
- **Scikit-learn**: Các thuật toán Machine Learning truyền thống
- **TensorFlow/Keras**: Deep Learning models (LSTM, CNN, Transformer)
- **NLTK**: Xử lý ngôn ngữ tự nhiên
- **Matplotlib & Seaborn**: Trực quan hóa dữ liệu

## 📁 Cấu trúc dự án

```
sentiment-analysis-ml/
│
├── data/                   # Thư mục chứa dữ liệu
│   ├── raw/               # Dữ liệu thô
│   ├── processed/         # Dữ liệu đã xử lý
│   └── README.md          # Hướng dẫn về dữ liệu
│
├── src/                    # Mã nguồn chính
│   ├── __init__.py
│   ├── preprocessing.py   # Tiền xử lý dữ liệu
│   ├── model.py          # Định nghĩa mô hình
│   ├── train.py          # Training script
│   └── predict.py        # Prediction script
│
├── notebooks/             # Jupyter notebooks
│   └── sentiment_analysis_demo.ipynb
│
├── tests/                 # Unit tests
│   └── test_preprocessing.py
│
├── requirements.txt       # Dependencies
├── LICENSE               # Giấy phép
└── README.md             # Tài liệu này
```

## 🚀 Hướng dẫn cài đặt

### 1. Clone repository

```bash
git clone https://github.com/RK-goldengate-co/sentiment-analysis-ml.git
cd sentiment-analysis-ml
```

### 2. Tạo môi trường ảo (khuyến nghị)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Tải dữ liệu NLTK cần thiết

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 💻 Hướng dẫn sử dụng

### Phân tích cảm xúc cơ bản

```python
from src.predict import SentimentAnalyzer

# Khởi tạo analyzer
analyzer = SentimentAnalyzer()

# Phân tích văn bản
text = "This product is amazing! I love it."
result = analyzer.predict(text)

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Training mô hình mới

```bash
python src/train.py --data data/processed/train.csv --epochs 10 --batch-size 32
```

### Sử dụng Jupyter Notebook

```bash
jupyter notebook notebooks/sentiment_analysis_demo.ipynb
```

## 📊 Kết quả

- **Accuracy**: ~85-90% trên tập test
- **F1-Score**: ~0.87
- **Thời gian inference**: < 100ms/sample

## 🤝 Đóng góp

Chúng tôi hoan nghênh mọi đóng góp! Vui lòng:

1. Fork repository
2. Tạo branch mới (`git checkout -b feature/AmazingFeature`)
3. Commit thay đổi (`git commit -m 'Add some AmazingFeature'`)
4. Push lên branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📝 License

Dự án được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.

## 📧 Liên hệ

Nếu có bất kỳ câu hỏi nào, vui lòng mở issue hoặc liên hệ qua email.

## 🙏 Acknowledgments

- Cảm ơn các thư viện mã nguồn mở đã hỗ trợ dự án này
- Dữ liệu training từ các dataset công khai (IMDB, Twitter Sentiment, etc.)
