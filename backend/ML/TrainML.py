import numpy as np
import pandas as pd
import os
import json
import joblib  # [필수]
import traceback
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM

# 버전 호환성 클래스
class SafeLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None) 
        super().__init__(*args, **kwargs)
    @classmethod
    def from_config(cls, config):
        config.pop('time_major', None)
        return super().from_config(config)

def run_training(sample_ids=None, dataset_path="backend/ML/ransomwaredataset.csv"):
    results = {
        "autoencoder": {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": [], "threshold": 0},
        "lstm": {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
    }

    # 경로 설정
    ML_DIR = os.path.join(settings.BASE_DIR, "backend", "ML")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(ML_DIR, "ransomwaredataset.csv")
    
    SCALER_PATH = os.path.join(ML_DIR, "scaler.pkl")
    AE_MODEL_PATH = os.path.join(ML_DIR, "autoencoder_model.h5")
    LSTM_MODEL_PATH = os.path.join(ML_DIR, "lstm_model.h5")
    AE_THRESHOLD_PATH = os.path.join(ML_DIR, "ae_threshold.json")

    # 1. 데이터 로드 및 필터링
    try:
        full_df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"[Error] 데이터 로드 실패: {e}")
        return results

    if sample_ids:
        # 문자열로 변환하여 매칭
        target_df = full_df[full_df["sample_id"].astype(str).isin([str(s) for s in sample_ids])]
    else:
        target_df = full_df
        
    if target_df.empty: return results

    # 정답 라벨 및 raw 데이터 분리
    drop_cols = ['sample_id', 'class_id', 'class_name']
    y_target = target_df["class_id"].tolist() if "class_id" in target_df.columns else [0] * len(target_df)
    X_raw = target_df.drop(columns=[c for c in drop_cols if c in target_df.columns])

    # 2. [핵심] 스케일러 로드 및 변환
    if not os.path.exists(SCALER_PATH):
        print("[CRITICAL] scaler.pkl 없음! train_standalone.py를 먼저 실행하세요.")
        return results
    
    try:
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X_raw) # 10개든 1개든 완벽하게 변환됨
    except Exception as e:
        print(f"[Error] 스케일링 실패: {e}")
        return results

    # ---------------------------------------------------------
    # [Autoencoder 분석]
    # ---------------------------------------------------------
    if os.path.exists(AE_MODEL_PATH):
        try:
            ae = load_model(AE_MODEL_PATH, compile=False)
            
            # 임계값 로드
            threshold = 0.05
            if os.path.exists(AE_THRESHOLD_PATH):
                with open(AE_THRESHOLD_PATH, 'r') as f:
                    threshold = json.load(f).get("threshold", 0.05)

            # 차원 안전장치
            if X_scaled.shape[1] != ae.input_shape[-1]:
                X_ae_in = X_scaled[:, :ae.input_shape[-1]]
            else:
                X_ae_in = X_scaled

            preds = ae.predict(X_ae_in, verbose=0)
            mse = np.mean(np.power(X_ae_in - preds, 2), axis=1)
            
            # 정확도(맞춤 여부) 계산
            acc_list = []
            for i, err in enumerate(mse):
                is_malware_pred = 1 if err > threshold else 0
                is_malware_true = 1 if y_target[i] > 0 else 0
                acc_list.append(1.0 if is_malware_pred == is_malware_true else 0.0)

            results["autoencoder"] = {
                "loss": mse.tolist(),
                "val_loss": [threshold] * len(mse),
                "accuracy": acc_list,
                "val_accuracy": [np.mean(acc_list)] * len(mse),
                "threshold": threshold
            }
        except Exception as e:
            print(f"[AE Error] {e}")

    # ---------------------------------------------------------
    # [LSTM 분석]
    # ---------------------------------------------------------
    if os.path.exists(LSTM_MODEL_PATH):
        try:
            lstm_model = load_model(LSTM_MODEL_PATH, custom_objects={'LSTM': SafeLSTM}, compile=False)
            
            # 차원 확장
            X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            lstm_preds = lstm_model.predict(X_lstm, verbose=0)
            
            # 1. 악성일 확률 (Confidence)
            if lstm_preds.shape[-1] > 1:
                final_probs = (1.0 - lstm_preds[:, 0]).tolist() 
                pred_classes = np.argmax(lstm_preds, axis=1)
            else:
                final_probs = lstm_preds.flatten().tolist()
                pred_classes = [1 if p > 0.5 else 0 for p in final_probs]

            # 2. 정답 여부 체크 (맞춤=1.0, 틀림=0.0)
            lstm_is_correct_list = []
            for pred, true_y in zip(pred_classes, y_target):
                true_bin = 1 if true_y > 0 else 0
                pred_bin = 1 if pred > 0 else 0
                lstm_is_correct_list.append(1.0 if true_bin == pred_bin else 0.0)

            # 3. [핵심] 평균 정확도 계산
            # 10개 중 9개 맞췄으면 0.9
            if len(lstm_is_correct_list) > 0:
                avg_acc = sum(lstm_is_correct_list) / len(lstm_is_correct_list)
            else:
                avg_acc = 0.0

            results["lstm"] = {
                # 파란선: 모델이 생각하는 '악성일 확률' (높을수록 위험)
                "accuracy": final_probs,      
                
                # 하늘색선: [수정됨] 이번 테스트의 '평균 정확도' (높을수록 모델이 똑똑함)
                "val_accuracy": [avg_acc] * len(final_probs),
                
                # 빨간선: 예측 오차 (정답과 예측확률의 거리, 낮을수록 좋음)
                # 정답(1) - 예측(0.9) = 0.1 (오차 작음)
                # 정답(1) - 예측(0.2) = 0.8 (오차 큼)
                "loss": [abs((1 if y > 0 else 0) - prob) for y, prob in zip(y_target, final_probs)],
                
                # 주황선: 목표 오차 (0.1 이하면 훌륭하다는 기준선)
                "val_loss": [0.1] * len(final_probs)
            }
            
            print(f"[Debug] LSTM 평균 정확도: {avg_acc * 100:.2f}%")

        except Exception as e:
            print(f"[LSTM Error] {e}")

    return results