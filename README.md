# Hệ thống phân tích CV tự động kết hợp XGBoost + SHAP để đánh giá mức độ phù hợp của ứng viên và trả về feedback có thể giải thích được (Explainable AI)
<pre>
1. Yêu cầu cài đặt
pip install fastapi uvicorn flask flask-cors joblib numpy shap xgboost scikit-learn

2. Khởi động – 3 Terminal

Terminal 1
cd "Đường dẫn file"
python3 server.py

Terminal 2
cd "Đường dẫn file"
python3 backend.py

Terminal 3
cd "Đường dẫn file"
python3 -m http.server 5500

Mở web tại:
http://127.0.0.1:5500/index.html
</pre>
<pre>
3. Cấu trúc dự án

fixed 7/
├── index.html                        # Entry point – frontend single-page app
├── app.js                            # Controller chính, quản lý AppState & điều hướng màn hình
├── styles.css                        # Global CSS
│
├── backend.py                        # FastAPI – AI engine chính (XGBoost + SHAP)
├── server.py                         # Flask – server phụ (lưu CV JSON, tiền xử lý TF-IDF)
│
├── components/
│   ├── Screen1_Transparency.js       # Màn 1: Consent & thông tin minh bạch
│   ├── Screen2_CVPreview.js          # Màn 2: Upload PDF + Claude extract CV data
│   ├── Screen3_Processing.js         # Màn 3: Gọi AI phân tích, hiển thị loading
│   ├── Screen4_Results.js            # Màn 4: Hiển thị kết quả SHAP + feedback
│   └── Screen5_Survey.js             # Màn 5: Khảo sát trải nghiệm người dùng
│
├── services/
│   ├── apiService.js                 # Gọi backend.py (FastAPI /predict)
│   ├── cvAnalysisService.js          # Gọi Claude API extract CV + gọi analysis
│   └── pdfService.js                 # Đọc PDF bằng pdf.js (client-side)
│
├── data/
│   ├── feature_names.json            # Danh sách feature names cho frontend
│   ├── mockData.js                   # Mock data phục vụ demo offline
│   └── mocks/                        # Sample CV & analysis JSON
│
├── utils/helpers.js                  # Utility functions (toast, escapeHtml, ...)
├── config/config.js                  # Cấu hình URL, debug flag, v.v.
│
├── xgb_model.pkl                     # Model XGBoost đã train
├── tfidf_vectorizer.pkl              # TF-IDF vectorizer
├── feature_names_fair.pkl            # Feature names (bias-filtered) dùng cho backend.py
├── feature_names_full.pkl            # Feature names đầy đủ dùng cho server.py
├── X_processed_full.npz              # Dataset đã xử lý (sparse matrix)
├── y_processed_full.npy              # Labels tương ứng
├── dataset1_.csv                     # Dataset gốc
│
├── Modelxgb.py                       # Script train XGBoost từ đầu
├── dataset 1 ana.py                  # Script EDA / phân tích dataset
│
└── output/                           # CV JSON được lưu sau mỗi lần phân tích
</pre>
<pre>
4. Flow
[Screen 1] Ứng viên đồng ý điều khoản minh bạch
     ↓
[Screen 2] Upload PDF → pdf.js đọc text → Claude AI extract
           thành JSON có cấu trúc (basicInfo, skills, experience, ...)
     ↓
[Screen 3] Ứng viên xác nhận CV → gọi backend.py /predict
           • TF-IDF vectorize text
           • XGBoost dự đoán match score
           • SHAP TreeExplainer giải thích top features
           • build_human_feedback()
     ↓
[Screen 4] Hiển thị kết quả:
           • Match score (gauge)
           • Strengths (top SHAP positives)
           • Development areas (top SHAP negatives)
           • SHAP base score
     ↓
[Screen 5] Khảo sát: Candidate đánh giá trải nghiệm
</pre>
<pre>
5. Chi tiết Screen 4

backend.py sử dụng shap.TreeExplainer để giải thích từng dự đoán:

explainer = shap.TreeExplainer(model)         # khởi tạo 1 lần
shap_values = explainer(input_data_dense)     # tính SHAP values

# Top 3 features đóng góp dương (strengths)
positives = sorted([x for x in impacts if x['impact'] > 0],
                   key=lambda x: x['impact'], reverse=True)[:3]

# Top 3 features đóng góp âm (development areas)
negatives = sorted([x for x in impacts if x['impact'] < 0],
                   key=lambda x: x['impact'])[:3]

Kết quả SHAP được build_human_feedback() dịch sang ngôn ngữ tự nhiên (tiếng Việt), lọc bỏ các demographic features (Gender_*, Race_*, Age_Scaled) để đảm bảo fairness.

Screen4_Results.js nhận object analysisResult từ AppState và render:
- strengths → list ✓ màu xanh
- developmentAreas → list màu đỏ
- explanation.base_score → SHAP base score của XGBoost
</pre>
<pre>
6. Output

Mỗi lần phân tích CV, file JSON được lưu tại output/:
output/cv_<TenUngVien>_<YYYYMMDD_HHMMSS>.json

7. Notes

- Model được train trên dataset1_.csv bằng Modelxgb.py
- feature_names_fair.pkl đã loại bỏ demographic features khỏi TF-IDF pool (dùng cho backend.py)
- feature_names_full.pkl giữ toàn bộ features (dùng cho server.py)
- SHAP chỉ chạy trên backend.py (FastAPI)
- server.py (Flask) không dùng SHAP
<pre>
