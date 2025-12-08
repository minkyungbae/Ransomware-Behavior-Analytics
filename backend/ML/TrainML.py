'''
[이 파일 속 코드의 목적]
-> ransomware-model.ipynb 파일 속 코드를 참고하여 Django에서 사용할 수 있도록한 파일입니다.

[포함된 기능 목록]
1. Autoencoder + LSTM 두 모델 모두 포함
2. 훈련 결과(loss/acc 그래프 데이터) JSON 형태로 반환
3. class_id 필터링 가능
4. Django View에서 import 후 바로 호출 가
'''

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# ==============================
# 1. 데이터 로드
# ==============================

def load_dataset(class_id=None, path="dataset.csv"):
    """
    CSV 데이터셋을 로드하고 class_id로 필터링한 뒤 반환.
    """
    df = pd.read_csv(path)

    if class_id is not None:
        df = df[df["class_id"] == class_id]

    # y 분리
    y = df["class_id"]
    X = df.drop(columns=["class_id"])

    return X, y


# ==============================
# 2. Autoencoder 모델 정의
# ==============================
def build_autoencoder(input_dim):
    """
    단층 Autoencoder 생성
    """
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(32, activation="relu")(input_layer)
    decoded = Dense(input_dim, activation="sigmoid")(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    return autoencoder


# ==============================
# 3. LSTM 모델 정의
# ==============================
def build_lstm(input_shape):
    """
    LSTM 기반 시계열 분류 모델
    input_shape = (timesteps, features)
    """
    model = tf.keras.Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ==============================
# 전체 훈련 프로세스
# (Django View에서 호출할 함수)
# ==============================
def run_training(class_id=None, dataset_path="dataset.csv"):
    """
    Notebook 훈련 코드를 기반으로
    Django가 호출 가능한 함수로 재구조화한 training pipeline.
    """

    # -------------------------------
    # Step1: 데이터 로드
    # -------------------------------
    X, y = load_dataset(class_id, dataset_path)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    input_dim = X_scaled.shape[1]

    # -------------------------------
    # Step2: Autoencoder 훈련
    # -------------------------------
    autoencoder = build_autoencoder(input_dim)

    history_ae = autoencoder.fit(
        X_scaled,
        X_scaled,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    # -------------------------------
    # Step3: LSTM용 입력 reshape
    # LSTM 입력: (samples, timesteps, features)
    # timesteps = 1 로 flatten 처리
    # -------------------------------
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    lstm_model = build_lstm((1, input_dim))

    history_lstm = lstm_model.fit(
        X_lstm,
        y,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=0
    )

    # -------------------------------
    # Step4: 결과 정리 후 반환
    # UI에서 그래프로 표시할 수 있게 배열로 반환
    # -------------------------------
    results = {
        "autoencoder": {
            "loss": history_ae.history["loss"],
            "val_loss": history_ae.history["val_loss"],
            "accuracy": history_ae.history["accuracy"],
            "val_accuracy": history_ae.history["val_accuracy"],
        },
        "lstm": {
            "loss": history_lstm.history["loss"],
            "val_loss": history_lstm.history["val_loss"],
            "accuracy": history_lstm.history["accuracy"],
            "val_accuracy": history_lstm.history["val_accuracy"],
        }
    }

    return results