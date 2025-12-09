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


# ==========================
# 데이터 로드
# ==========================
def load_dataset(class_ids=None, path="backend/ML/dataset.csv"):
    df = pd.read_csv(path)

    if "class_id" not in df.columns:
        raise KeyError("'class_id' 컬럼이 backend/ML/dataset.csv 안에 없습니다.")

    if class_ids is not None and len(class_ids) >0:
        df = df[df["class_id"].isin(class_ids)] # 다중 필터링(.init)

        if len(df) == 0:
            raise ValueError(f"class_id={class_ids} 에 해당하는 데이터가 없습니다.")

    y = df["class_id"]
    X = df.drop(columns=["class_id"])

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


def run_training(class_ids=None, dataset_path="backend/ML/dataset.csv", stream_callback=None):
    """
    Django View에서 호출되는 전체 training pipeline
    stream_callback: StreamingResponse로 로그를 보내기 위한 optional 함수
    """

    # ========== 내부 로그 함수 ========== 
    def log(msg):
        print(msg)  # 서버 콘솔에도 출력
        if stream_callback:
            stream_callback(msg)

    # -------------------------------
    # Step1: 데이터 로드
    # -------------------------------
    log(f"{class_ids} 데이터 로드 중...")
    X, y = load_dataset(class_ids, dataset_path)
    log(f"데이터 로드 완료 — 샘플 수: {len(X)}, feature: {X.shape[1]}")

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    input_dim = X_scaled.shape[1]

    # -------------------------------
    # Step2: Autoencoder 훈련
    # -------------------------------
    log("Autoencoder 모델 구성 중...")
    autoencoder = build_autoencoder(input_dim)
    log("Autoencoder 학습 시작...")

    history_ae = autoencoder.fit(
        X_scaled,
        X_scaled,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: log(
                    f"[AE] Epoch {epoch+1}/20 — loss={logs['loss']:.4f}, val_loss={logs['val_loss']:.4f}"
                )
            )
        ]
    )

    log("Autoencoder 학습 완료")

    # -------------------------------
    # Step3: LSTM 입력 reshape
    # -------------------------------
    log("LSTM 입력 형태 변환 중...")
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    lstm_model = build_lstm((1, input_dim))

    # -------------------------------
    # Step4: LSTM 훈련
    # -------------------------------
    log("LSTM 학습 시작...")

    history_lstm = lstm_model.fit(
        X_lstm,
        y,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: log(
                    f"[LSTM] Epoch {epoch+1}/20 — loss={logs['loss']:.4f}, val_loss={logs['val_loss']:.4f}"
                )
            )
        ]
    )

    log("LSTM 학습 완료")

    # -------------------------------
    # Step5: 결과 구조화 후 반환
    # -------------------------------
    log("훈련 결과 정리 중...")

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

    log("학습 종료")

    return results
