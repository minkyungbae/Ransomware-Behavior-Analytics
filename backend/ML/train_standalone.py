import numpy as np
import pandas as pd
import tensorflow as tf
import json
import os
import joblib 

from tensorflow.keras.models import Model, save_model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------
# [설정] 경로 정의 (cleaned_dataset.csv 사용)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "cleaned_dataset.csv")
AE_MODEL_PATH = os.path.join(BASE_DIR, "autoencoder_model.h5")
LSTM_MODEL_PATH = os.path.join(BASE_DIR, "lstm_model.h5")
AE_THRESHOLD_PATH = os.path.join(BASE_DIR, "ae_threshold.json")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

HISTORY_PATH = os.path.join(BASE_DIR, "training_history.json")

# ---------------------------------------------------------
# [모델] 아키텍처 개선 (Dropout 추가됨)
# ---------------------------------------------------------
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    
    # 인코더
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dropout(0.2)(encoded) # 과적합 방지
    encoded = Dense(32, activation='relu')(encoded)
    
    # 디코더
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dropout(0.2)(decoded) # 과적합 방지
    decoded = Dense(64, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape))
    model.add(Dropout(0.2)) # 과적합 방지
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2)) # 과적합 방지
    model.add(Dense(10, activation='softmax')) 
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ---------------------------------------------------------
# [계산] 임계값 저장
# ---------------------------------------------------------
def calculate_and_save_threshold(ae_model, X_normal_data):
    X_rec = ae_model.predict(X_normal_data, verbose=0)
    mse = np.mean(np.power(X_normal_data - X_rec, 2), axis=1)
    threshold = float(np.mean(mse) + 2 * np.std(mse))
    counts, bins = np.histogram(mse, bins=30)

    hist_data = {
        "threshold": threshold,
        "hist_counts": counts.tolist(),
        "hist_bins": bins.tolist()
    }
    
    with open(AE_THRESHOLD_PATH, 'w') as f:
        json.dump(hist_data, f)
    return threshold

# ---------------------------------------------------------
# [실행] 메인 훈련
# ---------------------------------------------------------
def train_and_save():
    print(f"📂 클렌징된 데이터 로드: {DATASET_PATH}")
    if not os.path.exists(DATASET_PATH):
        print("❌ cleaned_dataset.csv가 없습니다! clean_data.py를 먼저 실행하세요.")
        return

    df = pd.read_csv(DATASET_PATH)
    
    drop_cols = ['sample_id', 'class_id', 'class_name']
    existing_drop = [c for c in drop_cols if c in df.columns]
    X_raw = df.drop(columns=existing_drop).values
    Y = df['class_id'].values 

    # 1. 스케일러 학습 & 저장
    print("⚖️ 스케일러 학습 및 저장...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)
    joblib.dump(scaler, SCALER_PATH)

    input_dim = X_scaled.shape[1]

    # 2. Autoencoder 훈련 (Epoch 50으로 증가)
    print("\n🤖 Autoencoder 훈련 (Epoch 50)...")
    X_normal = X_scaled[Y == 0] if len(X_scaled[Y==0]) > 0 else X_scaled
    X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)

    ae = build_autoencoder(input_dim)
    history_ae = ae.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_val, X_val), verbose=0)
    
    thresh = calculate_and_save_threshold(ae, X_train)
    save_model(ae, AE_MODEL_PATH)
    print(f"✅ Autoencoder 저장 완료 (Threshold: {thresh:.4f})")

    # 3. LSTM 훈련 (Epoch 50으로 증가)
    print("\n🧠 LSTM 훈련 (Epoch 50)...")
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, input_dim))
    lstm = build_lstm((1, input_dim))
    history_lstm = lstm.fit(X_lstm, Y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    
    save_model(lstm, LSTM_MODEL_PATH)


    # 4. [추가] 훈련 기록(History) 저장
    history_data = {
        "autoencoder": history_ae.history,
        "lstm": history_lstm.history
    }
    # JSON 직렬화를 위해 float32 등을 float으로 변환할 필요가 있을 수 있음 (기본적으로 list of floats임)
    with open(HISTORY_PATH, 'w') as f:
        json.dump(history_data, f)
    print(f"✅ 훈련 히스토리 저장 완료: {HISTORY_PATH}")



if __name__ == "__main__":
    train_and_save()