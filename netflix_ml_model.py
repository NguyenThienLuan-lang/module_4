"""
Dự án: Xây dựng Machine Learning Model cho dữ liệu Netflix
Chuẩn quy trình 5 bước Scikit-Learn:
1. Load    -> pd.read_csv()
2. Split   -> train_test_split()
3. Fit     -> model.fit(X, y)
4. Predict -> model.predict(X)
5. Score   -> model.score(X, y)
"""

import sys

# Đảm bảo in tiếng Việt trên console Windows không bị lỗi UnicodeEncodeError
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

VALID_CLASSES = ['Adults (18+)', 'Kids & Family', 'Teens (13-14+)']

def map_audience_segment(rating):
    """Quy chuẩn mã rating thành 3 phân khúc khán giả."""
    if rating in ['TV-MA', 'R', 'NC-17']:
        return 'Adults (18+)'
    elif rating in ['TV-14', 'PG-13']:
        return 'Teens (13-14+)'
    elif rating in ['TV-PG', 'PG', 'TV-Y7', 'TV-Y', 'TV-G', 'G', 'TV-Y7-FV']:
        return 'Kids & Family'
    else:
        return 'Other'

def load_data(csv_path="netflix_titles_cleaned.csv"):
    """BƯỚC 1: LOAD - Đọc dữ liệu và chuẩn bị X, y."""
    df = pd.read_csv(csv_path)
    df['target_segment'] = df['rating'].apply(map_audience_segment)
    df_model = df[df['target_segment'] != 'Other'].copy()
    
    # Kết hợp type + listed_in + description tạo đặc trưng văn bản
    df_model['features'] = (
        df_model['type'] + " | " + 
        df_model['listed_in'] + " | " + 
        df_model['description']
    )
    
    X = df_model['features']
    y = df_model['target_segment']
    return X, y, df_model

def split_data(X, y, test_size=0.20, random_state=42):
    """BƯỚC 2: SPLIT - Tách tập train và test."""
    return train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )

def build_pipeline():
    """Tạo Scikit-Learn Pipeline: TF-IDF Vectorizer + Logistic Regression."""
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            stop_words='english',
            max_features=8000,
            ngram_range=(1, 2)
        )),
        ('classifier', LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=2.0
        ))
    ])

def train_model(X_train, y_train):
    """BƯỚC 3: FIT - Huấn luyện mô hình."""
    model = build_pipeline()
    model.fit(X_train, y_train)
    return model

def main():
    print("=" * 68)
    print("   QUY TRÌNH 5 BƯỚC XÂY DỰNG MODEL MACHINE LEARNING (SCIKIT-LEARN)")
    print("=" * 68)

    # =========================================================================
    # BƯỚC 1: LOAD (pd.read_csv)
    # =========================================================================
    print("\n>>> BƯỚC 1: LOAD DATA [pd.read_csv()]")
    csv_file = "netflix_titles_cleaned.csv"
    X, y, df_model = load_data(csv_file)
    print(f"[+] Đã load thành công file '{csv_file}'.")
    print(f"[+] Số lượng mẫu hợp lệ sau khi làm sạch: {len(df_model)} phim.")
    print("[+] Phân bố các nhóm độ tuổi trong tập dữ liệu:")
    for segment, count in y.value_counts().items():
        pct = (count / len(y)) * 100
        print(f"    - {segment:<15}: {count:>4} phim ({pct:.1f}%)")

    # =========================================================================
    # BƯỚC 2: SPLIT (train_test_split)
    # =========================================================================
    print("\n>>> BƯỚC 2: SPLIT DATA [train_test_split()]")
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.20, random_state=42)
    print(f"[+] Tập huấn luyện (X_train): {len(X_train)} mẫu (80%)")
    print(f"[+] Tập kiểm tra   (X_test) : {len(X_test)} mẫu (20%)")

    # =========================================================================
    # BƯỚC 3: FIT (model.fit)
    # =========================================================================
    print("\n>>> BƯỚC 3: FIT MODEL [model.fit(X, y)]")
    print("[*] Đang khởi tạo Pipeline: TF-IDF Vectorizer + Logistic Regression Classifier...")
    model = train_model(X_train, y_train)
    print("[+] Huấn luyện (Fit) hoàn tất thành công!")

    # Trích xuất xu hướng từ khóa nội dung
    print("\n[*] XU HƯỚNG TỪ KHÓA ĐẶC TRƯNG TỪ NỘI DUNG (Film Trends & Keywords):")
    tfidf = model.named_steps['tfidf']
    clf = model.named_steps['classifier']
    feature_names = tfidf.get_feature_names_out()
    for i, cls in enumerate(clf.classes_):
        top_indices = clf.coef_[i].argsort()[-6:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        print(f"    - Nhóm [{cls}]: {', '.join(top_words)}")

    # =========================================================================
    # BƯỚC 4: PREDICT (model.predict)
    # =========================================================================
    print("\n>>> BƯỚC 4: PREDICT [model.predict(X)]")
    y_pred = model.predict(X_test)
    print(f"[+] Đã thực hiện dự đoán xong cho {len(X_test)} mẫu trong tập Test.")

    print("\n[*] THỬ NGHIỆM DỰ ĐOÁN VỚI 3 TÁC PHẨM MỚI (INFERENCE):")
    new_samples = [
        {
            "title": "Little Animal Friends",
            "type": "Movie",
            "genre": "Children & Family Movies",
            "desc": "A cute pony and her adorable animal friends go on a magical musical quest to save the enchanted forest."
        },
        {
            "title": "Undercover Narcos",
            "type": "TV Show",
            "genre": "Crime TV Shows, Action & Adventure",
            "desc": "A ruthless cartel boss wages a violent and bloody turf war against federal agents and rival gangs."
        },
        {
            "title": "High School Romance",
            "type": "Movie",
            "genre": "Comedies, Romantic Movies",
            "desc": "Two high school students navigate the ups and downs of teenage love, prom night, and awkward friendships."
        }
    ]

    new_features = [f"{item['type']} | {item['genre']} | {item['desc']}" for item in new_samples]
    new_preds = model.predict(new_features)
    new_probs = model.predict_proba(new_features)

    for item, pred, probs in zip(new_samples, new_preds, new_probs):
        confidence = np.max(probs) * 100
        print(f"    - Phim: '{item['title']}' ({item['genre']})")
        print(f"      => Dự đoán: {pred} (Độ tin cậy: {confidence:.1f}%)")

    # =========================================================================
    # BƯỚC 5: SCORE (model.score)
    # =========================================================================
    print("\n>>> BƯỚC 5: SCORE [model.score(X, y)]")
    accuracy = model.score(X_test, y_test)
    print(f"[+] Điểm số chính xác (Mean Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\n[*] BẢNG BÁO CÁO PHÂN LOẠI CHI TIẾT (Classification Report):")
    print(classification_report(y_test, y_pred, digits=4))

    print("[*] MA TRẬN NHẦM LẪN (Confusion Matrix):")
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    cm_df = pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_)
    print(cm_df)

    print("\n" + "=" * 68)
    print("   HOÀN THÀNH QUY TRÌNH 5 BƯỚC THEO CHUẨN SCIKIT-LEARN!")
    print("=" * 68)

if __name__ == "__main__":
    main()
