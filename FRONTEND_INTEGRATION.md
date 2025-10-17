# Frontend Integration Guide

## Tích hợp Frontend React với Backend FastAPI

Tài liệu này hướng dẫn cách tích hợp giao diện React frontend với backend FastAPI để tạo hệ thống phân tích sentiment hoàn chỉnh.

## 📋 Cấu trúc Frontend

```
frontend/
├── public/
│   └── index.html          # HTML template
├── src/
│   ├── App.js              # Component chính
│   ├── App.css             # Styles
│   └── index.js            # Entry point
└── package.json            # Dependencies
```

## 🚀 Cài đặt và Chạy Frontend

### 1. Cài đặt dependencies

```bash
cd frontend
npm install
```

### 2. Chạy development server

```bash
npm start
```

Ứng dụng sẽ chạy tại `http://localhost:3000`

### 3. Build cho production

```bash
npm run build
```

Files sẽ được tạo trong thư mục `build/`

## 🔧 Cấu hình API URL

Frontend kết nối với backend thông qua biến môi trường. Tạo file `.env` trong thư mục `frontend/`:

```env
REACT_APP_API_URL=http://localhost:8000
```

Đối với production, thay đổi URL thành địa chỉ backend thực tế:

```env
REACT_APP_API_URL=https://your-api-domain.com
```

## 🔌 API Endpoints được sử dụng

### 1. Predict Sentiment

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "text": "Your text to analyze"
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.95,
  "text": "Your text to analyze"
}
```

## 🎨 Tính năng Frontend

### 1. Nhập văn bản
- Textarea để người dùng nhập văn bản cần phân tích
- Validation để đảm bảo văn bản không rỗng

### 2. Phân tích Sentiment
- Button để gửi request đến API
- Loading state trong khi xử lý
- Error handling cho các lỗi network hoặc API

### 3. Hiển thị kết quả
- Sentiment: positive/negative/neutral
- Confidence score với thanh progress bar
- Color coding theo loại sentiment:
  - 🟢 Positive: Green (#4CAF50)
  - 🔴 Negative: Red (#f44336)
  - 🟠 Neutral: Orange (#FF9800)

## 🔄 Workflow tích hợp

```
1. User nhập text vào textarea
   ↓
2. Click button "Analyze Sentiment"
   ↓
3. Frontend gửi POST request đến /predict
   ↓
4. Backend xử lý và trả về kết quả
   ↓
5. Frontend hiển thị sentiment và confidence
```

## 🐛 Error Handling

Frontend xử lý các trường hợp lỗi:

1. **Văn bản rỗng**: Hiển thị message yêu cầu nhập text
2. **API Error**: Hiển thị error message từ backend
3. **Network Error**: Hiển thị generic error message

## 🔐 CORS Configuration

Backend cần cấu hình CORS để cho phép frontend gọi API. Trong `app.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Đối với production, cập nhật `allow_origins` với domain thực tế:

```python
allow_origins=[
    "http://localhost:3000",
    "https://your-frontend-domain.com"
]
```

## 🚀 Deploy cùng nhau

### Option 1: Serve frontend từ FastAPI

```python
from fastapi.staticfiles import StaticFiles

# Build frontend trước
app.mount("/", StaticFiles(directory="frontend/build", html=True), name="static")
```

### Option 2: Deploy riêng biệt

- **Backend**: Deploy lên Heroku, AWS, GCP, etc.
- **Frontend**: Deploy lên Vercel, Netlify, etc.
- Cập nhật `REACT_APP_API_URL` với backend URL

## 📝 Testing

### Test manual workflow:

1. Chạy backend: `uvicorn app:app --reload`
2. Chạy frontend: `cd frontend && npm start`
3. Truy cập `http://localhost:3000`
4. Nhập text và test các scenarios:
   - Text positive
   - Text negative
   - Text neutral
   - Empty text
   - Very long text

## 🎯 Cải tiến tiếp theo

- [ ] Thêm batch analysis (nhiều texts cùng lúc)
- [ ] History của các analyses trước
- [ ] Export results to CSV/JSON
- [ ] Realtime analysis khi typing
- [ ] Multi-language support
- [ ] Authentication & user accounts

## 📚 Resources

- [React Documentation](https://react.dev/)
- [Axios Documentation](https://axios-http.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Create React App](https://create-react-app.dev/)

## 🤝 Đóng góp

Nếu có vấn đề hoặc suggestions, vui lòng tạo issue hoặc pull request!
