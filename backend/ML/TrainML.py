import numpy as np
import pandas as pd
import tensorflow as tf
import os

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# ==========================
# 데이터 로드
# ==========================
def load_dataset(target_ids=None, path="backend/ML/ransomwaredataset.csv"):
    # 파일 경로 예외처리 (os.path 사용 권장)
    if not os.path.exists(path):
        # 만약 경로가 안 맞으면 기본 경로 시도 (settings.BASE_DIR 없이 실행될 때 대비)
        path = "backend/ML/ransomwaredataset.csv"
        
    df = pd.read_csv(path)

    # 필터링 로직 수정 (sample_id 기준인지 class_id 기준인지 확인 필요)
    # views.py에서 sample_ids를 넘겨주므로, 여기서는 sample_id로 필터링하는 것이 안전함
    # 만약 CSV에 sample_id 컬럼이 있다면 아래 사용:
    filter_col = "sample_id" if "sample_id" in df.columns else "class_id"
    
    if target_ids is not None and len(target_ids) > 0:
        df = df[df[filter_col].isin(target_ids)]
        
        if len(df) == 0:
            # 빈 데이터면 에러 대신 빈 값 반환 (서버 다운 방지)
            print(f"Warning: {target_ids}에 해당하는 데이터가 없습니다.")
            return pd.DataFrame(), pd.Series()

    # 타겟 컬럼(y) 설정 (class_id가 정답 라벨이라고 가정)
    if "class_id" in df.columns:
        y = df["class_id"]
        # 학습에 불필요한 컬럼 제거 (에러 방지를 위해 errors='ignore' 추가)
        X = df.drop(columns=["class_id", "sample_id", "class_name"], errors='ignore')
    else:
        # 정답 컬럼이 없을 경우 (예외 처리)
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

    return X, y


# ==========================
# Autoencoder 구성
# ==========================
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(32, activation="relu")(input_layer)
    decoded = Dense(input_dim, activation="sigmoid")(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    return autoencoder


# ==========================
# LSTM 모델 구성
# ==========================
def build_lstm(input_shape):
    model = tf.keras.Sequential([
        # LSTM 레이어
        LSTM(32, return_sequences=False, input_shape=input_shape), # 64 -> 32로 줄임 (속도 향상)
        Dense(16, activation="relu"), # 32 -> 16으로 줄임 (속도 향상)
        Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def run_training(sample_ids=None, dataset_path="backend/ML/ransomwaredataset.csv"):
    """
    Django View에서 호출되는 전체 training pipeline
    """
    
    # [수정] 결과 딕셔너리 초기화
    results = {}

    print(f"[Info] 데이터 로드 중... (Target IDs: {sample_ids})")
    X, y = load_dataset(sample_ids, dataset_path)
    
    if X.empty:
        return {"error": "선택된 샘플에 대한 데이터가 없습니다."}

    print(f"[Info] 데이터 로드 완료 — 샘플 수: {len(X)}")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    input_dim = X_scaled.shape[1]

    # -------------------------------
    # Step2: Autoencoder 훈련 (속도 최적화 적용)
    # -------------------------------
    print("[Info] Autoencoder 학습 시작...")
    autoencoder = build_autoencoder(input_dim)

    history_ae = autoencoder.fit(
        X_scaled,
        X_scaled,
        epochs=5,         # [중요] 20 -> 5로 감소 (테스트용, 속도 4배 향상)
        batch_size=128,   # [중요] 32 -> 128로 증가 (병렬 처리량 증가)
        validation_split=0.2,
        verbose=0,
    )

    # 임계값 계산
    ae_predictions = autoencoder.predict(X_scaled, verbose=0)
    reconstruction_errors = np.mean(np.square(X_scaled - ae_predictions), axis=1)
    THRESHOLD = float(np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)) # float 형변환 (JSON 오류 방지)
    
    print(f"[Info] Autoencoder 학습 완료 (Threshold: {THRESHOLD:.4f})")

    # -------------------------------
    # Step3: LSTM 훈련 (속도 최적화 적용)
    # -------------------------------
    print("[Info] LSTM 학습 시작...")
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    lstm_model = build_lstm((1, input_dim))

    history_lstm = lstm_model.fit(
        X_lstm,
        y,
        epochs=5,         # [중요] 20 -> 5로 감소
        batch_size=128,   # [중요] 32 -> 128로 증가
        validation_split=0.2,
        verbose=0,
    )
    print("[Info] LSTM 학습 완료")

    # -------------------------------
    # Step5: 결과 반환 (수정됨: val_loss 포함)
    # -------------------------------
    results = {
        "autoencoder": {
            "loss": history_ae.history["loss"],
            "val_loss": history_ae.history.get("val_loss", []), # [추가] 그래프 그리기용
            "accuracy": history_ae.history["accuracy"],
            "val_accuracy": history_ae.history.get("val_accuracy", []), # [추가]
            "threshold": THRESHOLD,
        },
        "lstm": {
            "loss": history_lstm.history["loss"],
            "val_loss": history_lstm.history.get("val_loss", []), # [추가]
            "accuracy": history_lstm.history["accuracy"],
            "val_accuracy": history_lstm.history.get("val_accuracy", []), # [추가]
        }
    }

    return results