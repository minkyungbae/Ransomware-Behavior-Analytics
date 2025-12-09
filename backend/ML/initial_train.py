import numpy as np
import pandas as pd
import tensorflow as tf
import json
import os

from tensorflow.keras.models import Model, save_model
from tensorflow.keras.layers import Input, Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# ----------------------------------------------------
# 📌 파일 경로 정의
# 이 경로는 TrainML.py에 정의된 경로와 일치해야 합니다.
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "ransomwaredataset.csv") # 데이터셋 경로
AE_MODEL_PATH = os.path.join(BASE_DIR, "autoencoder_model.h5")
LSTM_MODEL_PATH = os.path.join(BASE_DIR, "lstm_model.h5")
AE_THRESHOLD_PATH = os.path.join(BASE_DIR, "ae_threshold.json")


# ====================================================
# A. 모델 아키텍처 정의 (notebook 기반으로 단순화)
# ====================================================

def build_autoencoder(input_dim):
    """Autoencoder 모델 정의"""
    input_layer = Input(shape=(input_dim,))
    # 인코더
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    # 디코더
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded) # 입력과 같은 차원
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_lstm(input_shape):
    """LSTM 모델 정의 (이진 분류)"""
    model = tf.keras.Sequential([
        LSTM(units=64, input_shape=input_shape),
        Dense(32, activation='relu'),
        Dense(10, activation='softmax') 
    ])
    # Autoencoder의 재구성 오차를 사용하지 않으므로, 일반적인 이진 분류 지표 사용
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# ====================================================
# B. 임계값 계산 및 저장 로직
# ====================================================

def calculate_and_save_threshold(ae_model, X_normal_data):
    """
    훈련된 Autoencoder를 사용하여 정상 데이터의 재구성 오차를 계산하고 임계값을 저장
    """
    print("\n[3] Autoencoder 임계값 계산 및 저장 중...")
    
    # 1. 재구성 오차 계산
    X_reconstructed = ae_model.predict(X_normal_data, verbose=0)
    reconstruction_errors = np.mean(np.square(X_normal_data - X_reconstructed), axis=1)

    # 2. 임계값 결정 (일반적인 방법: 평균 + 2 * 표준편차)
    mean_err = np.mean(reconstruction_errors)
    std_err = np.std(reconstruction_errors)
    # 보통 2~3 표준편차를 사용하며, 여기서는 2를 사용합니다.
    THRESHOLD = mean_err + (2 * std_err) 
    
    # 3. 임계값을 JSON 파일로 저장
    try:
        with open(AE_THRESHOLD_PATH, 'w') as f:
            json.dump({"threshold": float(THRESHOLD)}, f)
        print(f"✅ 임계값 {THRESHOLD:.6f}가 '{os.path.basename(AE_THRESHOLD_PATH)}'에 저장되었습니다.")
        return THRESHOLD
    except Exception as e:
        print(f"⚠️ 임계값 파일 저장 실패: {e}")
        return None

# ====================================================
# C. 메인 훈련 루틴
# ====================================================

def initial_train_and_save():
    print(f"데이터셋 로드: {DATASET_PATH}")
    
    try:
        df = pd.read_csv(DATASET_PATH)
    except FileNotFoundError:
        print(f"🚨 오류: {DATASET_PATH} 파일을 찾을 수 없습니다. 경로를 확인하십시오.")
        return

    # 1. 데이터셋 분리 및 전처리
    # 특징(X)과 레이블(Y) 분리. 'class_name'은 사용하지 않음.
    feature_cols = df.columns.drop(['sample_id', 'class_id', 'class_name']).tolist()
    X = df[feature_cols].values
    Y = df['class_id'].values # Y는 class_id (0:정상, 1~n:악성)

    # 스케일링
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    input_dim = X_scaled.shape[1]
    
    # 2. Autoencoder 훈련 (정상 데이터만 사용)
    # ⚠️ 중요: Autoencoder는 정상 데이터(class_id=0)만 사용하여 훈련해야 합니다.
    # CSV 파일 스니펫에는 class_id=0이 보이지 않으므로, 데이터셋에 정상 샘플이 있다고 가정합니다.
    print("\n[1] Autoencoder 훈련 (정상 데이터만 사용) 시작...")
    X_normal = X_scaled[Y == 0]
    
    if X_normal.shape[0] == 0:
        print("❌ 경고: 정상 데이터(class_id=0)가 발견되지 않았습니다. 전체 데이터를 사용합니다 (권장되지 않음).")
        X_ae_train = X_scaled
    else:
        X_ae_train = X_normal
        
    X_ae_train, X_ae_val = train_test_split(X_ae_train, test_size=0.2, random_state=42)
    
    ae_model = build_autoencoder(input_dim)
    ae_model.fit(X_ae_train, X_ae_train, epochs=50, batch_size=32, validation_data=(X_ae_val, X_ae_val), verbose=1)
    
    # 3. 임계값 계산 및 저장 (Autoencoder 훈련 직후)
    calculate_and_save_threshold(ae_model, X_ae_train)
    save_model(ae_model, AE_MODEL_PATH)
    print(f"✅ Autoencoder 모델이 '{os.path.basename(AE_MODEL_PATH)}'에 저장되었습니다.")


    # 4. LSTM 훈련 (전체 데이터 사용)
    print("\n[4] LSTM 훈련 (전체 데이터 사용) 시작...")
    # LSTM 입력 reshape: (샘플 수, 1, 피처 수)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, input_dim)) 
    
    # 이진 분류를 위해 레이블을 0 또는 1로 간주합니다.
    # class_id가 0(정상)이면 0, 그 외(1~n)는 1(악성)로 변환 (필요시)
    Y_binary = np.where(Y > 0, 1, Y) 
    
    lstm_model = build_lstm((1, input_dim))
    lstm_model.fit(X_lstm, Y_binary, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
    
    # 5. LSTM 모델 저장
    save_model(lstm_model, LSTM_MODEL_PATH)
    print(f"✅ LSTM 모델이 '{os.path.basename(LSTM_MODEL_PATH)}'에 저장되었습니다.")

if __name__ == "__main__":
    initial_train_and_save()