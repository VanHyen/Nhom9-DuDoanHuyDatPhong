"""Offline convenience training; the notebook is the canonical experiment record.

Explicit output directory required. Does not update the serving artifact, bounds,
or reported metrics. Never imported by the web application.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def train(output_dir):
    if output_dir.resolve() == (ROOT / "models").resolve():
        raise ValueError("Không ghi đè models/ đang phục vụ. Chọn thư mục xuất offline khác.")
    if (output_dir / "logistic_pipeline.pkl").exists():
        raise FileExistsError("Tệp đầu ra đã tồn tại; chọn thư mục mới.")
    import os
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    print("1. Đang tải dữ liệu gốc...")
    url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv"
    df = pd.read_csv(url)

    print("2. Làm sạch và loại bỏ các biến rò rỉ dữ liệu...")
    # Loại bỏ triệt để các biến phát sinh sau thời điểm đặt
    leakage_cols = [
        'reservation_status', 'reservation_status_date',
        'assigned_room_type', 'booking_changes', 'days_in_waiting_list'
    ]
    df_clean = df.drop(columns=leakage_cols)
    df_clean = df_clean[df_clean[['adults', 'children', 'babies']].sum(axis=1) > 0].copy()
    df_clean = df_clean[df_clean['adr'] >= 0].copy()

    # Xử lý missing
    df_clean['children'] = df_clean['children'].fillna(0)
    df_clean['country'] = df_clean['country'].fillna('Unknown')
    df_clean['has_agent'] = df_clean['agent'].notnull().astype(int)
    df_clean['has_company'] = df_clean['company'].notnull().astype(int)
    df_clean = df_clean.drop(columns=['agent', 'company'])

    # Tách biến mục tiêu
    X = df_clean.drop(columns=['is_canceled'])
    y = df_clean['is_canceled']

    print("3. Phân chia tập dữ liệu (70% Train)...")
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    # Danh sách đặc trưng
    num_cols = [
        'lead_time', 'arrival_date_year', 'arrival_date_week_number', 'arrival_date_day_of_month',
        'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
        'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
        'adr', 'required_car_parking_spaces', 'total_of_special_requests', 'has_agent', 'has_company'
    ]

    cat_cols = [
        'hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
        'distribution_channel', 'reserved_room_type', 'deposit_type', 'customer_type'
    ]

    print("4. Xây dựng Pipeline và huấn luyện mô hình Logistic Regression...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Huấn luyện mô hình
    full_pipeline.fit(X_train, y_train)

    # Lưu tệp mô hình đè vào thư mục models/
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "logistic_pipeline.pkl"
    joblib.dump(full_pipeline, model_path)

    print(f"Created offline model: {model_path}. Export matching bounds and model card before serving.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline training only; does not reproduce all notebook experiments")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train(args.output_dir)
