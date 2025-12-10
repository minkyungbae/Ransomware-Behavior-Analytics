import pandas as pd
import numpy as np
import os
from scipy import stats

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "ransomwaredataset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "cleaned_dataset.csv")

def clean_data():
    print(f"🧹 데이터 클렌징 시작: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        print("❌ 원본 데이터 파일이 없습니다.")
        return

    df = pd.read_csv(INPUT_PATH)
    original_len = len(df)

    # 1. 이상치 제거 (Z-score 3 이상 제거)
    # 숫자형 컬럼만 선택 (class_id 제외)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('class_id', errors='ignore')
    
    # Z-score 계산
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    # 모든 컬럼에서 Z-score가 3 미만인 행만 남김
    df_clean = df[(z_scores < 3).all(axis=1)]
    
    print(f"   - 이상치 제거: {original_len} -> {len(df_clean)}개 (삭제된 행: {original_len - len(df_clean)})")

    # 2. 클래스 균형 맞추기 (단순 오버샘플링 - 라이브러리 설치 없이 Pandas로 해결)
    # class_id가 0인 것(정상)과 0이 아닌 것(악성)의 비율 확인
    normal = df_clean[df_clean['class_id'] == 0]
    malware = df_clean[df_clean['class_id'] > 0]

    print(f"   - 균형 전: 정상 {len(normal)}개, 악성 {len(malware)}개")

    # 데이터가 적은 쪽을 많은 쪽 개수만큼 늘림 (복제)
    if len(normal) > len(malware):
        malware_upsampled = malware.sample(n=len(normal), replace=True, random_state=42)
        df_balanced = pd.concat([normal, malware_upsampled])
    else:
        normal_upsampled = normal.sample(n=len(malware), replace=True, random_state=42)
        df_balanced = pd.concat([normal_upsampled, malware])

    print(f"   - 균형 후 총 데이터: {len(df_balanced)}개")

    # 3. 저장
    df_balanced.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ 클렌징 완료! 저장된 파일: {OUTPUT_PATH}")

if __name__ == "__main__":
    clean_data()