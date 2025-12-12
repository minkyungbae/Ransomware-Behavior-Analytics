import numpy as np
import pandas as pd
import os
import json
import joblib  # [필수]
import traceback
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
import logging


# 로거 설정 (콘솔에만 출력하고 HTTP 응답에는 영향 안 줌)
logger = logging.getLogger(__name__)


# 버전 호환성 클래스
class SafeLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None) 
        super().__init__(*args, **kwargs)
    @classmethod
    def from_config(cls, config):
        config.pop('time_major', None)
        return super().from_config(config)
    

# =========================================
# 1. 헬퍼 함수 : Threshold 계산 및 해석 로직
# =========================================
def compute_sample_threshold(sample, recon_sample):
    """
    샘플 자체 오차 분포(feature-wise error)의 70% 분위를 
    개별 Threshold로 설정
    """
    errors = (sample-recon_sample)**2
    # 70percentile을 threshold로 사용
    return float(np.percentile(errors, 70))

def interpret_case(mse, th):
    """
    MSE와 Threshold 비율에 따른 상태 해석
    """
    if th == 0: return "계산 불가"
    ratio = mse / th
    if ratio < 0.5:
        return "정상 (Safe) - 정상 Encryptor 패턴과 매우 유사함"
    elif ratio < 0.8:
        return "정상 (Stable) - 정상 패턴 대비 미세한 변동 존재"
    elif ratio < 1.0:
        return "경계 (Warning) - 정상 범위이지만 편차가 다소 있음"
    elif ratio < 1.5:
        return "경미한 이상 (Minor Anomaly) - 정상 패턴과 일부 차이가 감지됨"
    elif ratio < 2.0:
        return "이상 (Anomaly) - 정상 Encryptor 행동 패턴과 명확하게 다름"
    else:
        return "고위험 (Critical) - baseline을 크게 벗어난 공격성 패턴"
    


# =========================================
# 2. 메인 실행 함수
# =========================================
def run_training(sample_ids=None, dataset_path="backend/ML/ransomwaredataset.csv"):
    results = {
        "autoencoder": {},
        "lstm": {},
        "predictions": []
    }

    # 경로 설정
    ML_DIR = os.path.join(settings.BASE_DIR, "backend", "ML")
    HISTORY_PATH = os.path.join(ML_DIR, "training_history.json")

    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(ML_DIR, "ransomwaredataset.csv")
    
    SCALER_PATH = os.path.join(ML_DIR, "scaler.pkl")
    AE_MODEL_PATH = os.path.join(ML_DIR, "autoencoder_model.h5")
    LSTM_MODEL_PATH = os.path.join(ML_DIR, "lstm_model.h5")
    AE_THRESHOLD_PATH = os.path.join(ML_DIR, "ae_threshold.json")

    # 1. 훈련 히스토리 로드 (그래프용)
    history_data = {"autoencoder": {}, "lstm": {}}
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, 'r') as f:
                history_data = json.load(f)
        except Exception as e:
            # print(f"[Warning] 히스토리 로드 실패: {e}")
            pass

    # 2. 데이터 로드 및 필터링
    try:
        full_df = pd.read_csv(dataset_path)
    except Exception as e:
        # print(f"[Error] 데이터 로드 실패: {e}")
        return results

    if sample_ids:
        # 문자열로 변환하여 매칭
        target_df = full_df[full_df["sample_id"].astype(str).isin([str(s) for s in sample_ids])]
    else:
        target_df = full_df
        
    if target_df.empty: return results

    # 정답 라벨 및 raw 데이터 분리
    drop_cols = ['sample_id', 'class_id', 'class_name']
    # class_id가 있으면 가져오고 없으면 0 처리
    y_target = target_df["class_id"].tolist() if "class_id" in target_df.columns else [0] * len(target_df)
    X_raw = target_df.drop(columns=[c for c in drop_cols if c in target_df.columns])

    # 3. [핵심] 스케일러 로드 및 변환
    if not os.path.exists(SCALER_PATH):
        # print("[CRITICAL] scaler.pkl 없음! train_standalone.py를 먼저 실행하세요.")
        return results
    
    try:
        scaler = joblib.load(SCALER_PATH)
        X_scaled = scaler.transform(X_raw.values) # 10개든 1개든 완벽하게 변환됨
    except Exception as e:
        # print(f"[Error] 스케일링 실패: {e}")
        return results



    # ---------------------------------------------------------
    # [Autoencoder 분석]
    # ---------------------------------------------------------
    global_threshold = 0.05
    hist_data = {"counts": [], "bins": []}

    if os.path.exists(AE_MODEL_PATH):
        try:
            ae = load_model(AE_MODEL_PATH, compile=False)
            
            # 글로벌 임계값 및 히스토그램 로드
            if os.path.exists(AE_THRESHOLD_PATH):
                with open(AE_THRESHOLD_PATH, 'r') as f:
                    loaded_data = json.load(f)
                    global_threshold = loaded_data.get("threshold", 0.05)
                    hist_data["counts"] = loaded_data.get("hist_counts", [])
                    hist_data["bins"] = loaded_data.get("hist_bins", [])


            # 차원 안전장치
            if X_scaled.shape[1] != ae.input_shape[-1]:
                X_ae_in = X_scaled[:, :ae.input_shape[-1]]
            else:
                X_ae_in = X_scaled

            
            # 예측 및 재구성
            recon_samples = ae.predict(X_ae_in, verbose=0)

            # AE 그래프 데이터 구성
            results["autoencoder"] = {
                "loss": history_data["autoencoder"].get("loss", []),
                "val_loss": history_data["autoencoder"].get("val_loss", []),
                "accuracy": history_data["autoencoder"].get("accuracy", []),
                "global_threshold": global_threshold,
                "histogram": hist_data
            }

        except Exception as e:
            # print(f"[AE Error] {e}")
            recon_samples = np.zeros_like(X_scaled)


            
    # ---------------------------------------------------------
    # [LSTM 분석 & 결과 통합]
    # ---------------------------------------------------------
    lstm_probs_list = []

    if os.path.exists(LSTM_MODEL_PATH):
        try:
            lstm_model = load_model(LSTM_MODEL_PATH, custom_objects={'LSTM': SafeLSTM}, compile=False)
            
            # 차원 확장
            X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            lstm_preds = lstm_model.predict(X_lstm, verbose=0)


            results["lstm"] = {
                "loss": history_data["lstm"].get("loss", []),
                "val_loss": history_data["lstm"].get("val_loss", []),
                "accuracy": history_data["lstm"].get("accuracy", []),  
                "val_accuracy": history_data["lstm"].get("val_accuracy", []),
            }

            lstm_probs_list = lstm_preds.tolist()


        except Exception as e:
            # print(f"[LSTM Error] {e}")
            pass



    # ---------------------------------------------------------
    # [데이터 병합 : 샘플별 상세 결과 생성]
    # ---------------------------------------------------------
    sample_to_class = {}
    if "class_name" in target_df.columns:
        sample_to_class = dict(zip(target_df["sample_id"].astype(str), target_df["class_name"]))

    predictions_list = []

    class_names = {
    0: "Encryptor",
    1: "Locker",
    2: "Wiper",
    3: "Worm-propagating Ransom",
    4: "Human-operated Ransom",
    5: "Phishing-based Ransom",
    6: "RDP Brute-force based",
    7: "Exploit-based Ransom",
    8: "USB/Removable-media Ransom",
    9: "Cloud/SaaS-targeted Ransom",
    }


    for i in range(len(target_df)):
            sample_id = str(target_df.iloc[i]["sample_id"])

            # 1. AE: Sample Specific Threshold & MSE
            sample_vec = X_scaled[i]
            recon_vec = recon_samples[i]

            errors = (sample_vec - recon_vec)**2
            mse_val = float(np.mean(errors))

            counts, bins = np.histogram(errors, bins=30)
            sample_th = compute_sample_threshold(sample_vec, recon_vec)
            interpretation = interpret_case(mse_val, sample_th)

            # Anomaly 여부 판단 (ratio >= 1.0이면 경계 이상)
            is_anomaly = (mse_val > sample_th)
            judgment = "Anomaly" if is_anomaly else "Normal"

            # 2. LSTM: Class Probabilities (10 classes)
            # 만약 LSTM 예측이 실패했으면 0으로 채움
            if i < len(lstm_probs_list):
                probs = lstm_probs_list[i]
            else:
                probs = [0.0] * 10


            # ⭐ 클래스 이름과 매핑
            class_probs = {class_names[j]: float(probs[j]) for j in range(10)}

            true_class_name = sample_to_class.get(sample_id, "정보없음")


            
            predictions_list.append({
            "sample_id": sample_id,
            "mse": mse_val,
            "threshold": sample_th,
            "is_anomaly": is_anomaly,
            "judgment": judgment,
            "interpretation": interpretation,
            "lstm_probs": probs,  # [0.1, 0.05, ..., 0.8] (10개)
            "class_probs": class_probs,
            "true_class": int(y_target[i]),
            "true_class_name": true_class_name,
            "histogram": {
                "counts": counts.tolist(),
                "bins": bins.tolist()
            }     
            })

    results["predictions"] = predictions_list

    return results

            