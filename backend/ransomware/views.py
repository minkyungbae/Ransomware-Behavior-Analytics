import json
import random
import time
import pandas as pd
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.shortcuts import render

from ML.TrainML import run_training

# 서버 시작 시 1회 로딩 (데이터가 크면 메모리 부족 주의)
dataset_path = settings.BASE_DIR / "backend" / "ML" / "ransomwaredataset.csv"
try:
    GLOBAL_DF = pd.read_csv(dataset_path)
    print("CSV Loaded into Memory (Success)")
except Exception as e:
    print(f"CSV Load Error: {e}")
    GLOBAL_DF = pd.DataFrame()

def mainpage(request):
    return render(request, "mainpage.html")


# ============================================================
# 1. sample_id 목록 제공 (수정됨)
# ============================================================
def get_samples(request):
    global GLOBAL_DF
    
    # 데이터가 없으면 에러 반환
    if GLOBAL_DF.empty:
        return JsonResponse({"error": "Dataset is empty or not loaded"}, status=500)

    # 1. 전체 고유 ID 리스트 추출
    all_sample_ids = GLOBAL_DF["sample_id"].unique().tolist()
    
    # 2. 랜덤 10개 추출
    target_count = 10
    if len(all_sample_ids) > target_count:
        sample_ids = random.sample(all_sample_ids, target_count)
    else:
        sample_ids = all_sample_ids
    
    sample_ids.sort()
    return JsonResponse({"sample_ids": sample_ids})


# ============================================================
# 2. 모델 훈련 API
# ============================================================
@csrf_exempt
def train_models(request):
    """
    이 함수는 'StreamingHttpResponse'를 사용하여
    프론트엔드에 진행 상황을 실시간으로 보냅니다.
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    # 요청 데이터 파싱
    try:
        body = json.loads(request.body)
        sample_ids = body.get("sample_ids", [])
    except:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if not sample_ids:
        return JsonResponse({"error": "sample_ids is required"}, status=400)

    # [수정 포인트] dataset_path를 함수 내부에서 명확하게 정의합니다.
    # 이렇게 하면 내부 함수(event_stream)가 이 변수를 확실하게 찾을 수 있습니다.
    csv_path_str = str(settings.BASE_DIR / "backend" / "ML" / "ransomwaredataset.csv")

    # [핵심] 제너레이터 함수 정의 (진행 상황을 한 줄씩 yield)
    def event_stream():
        # Step 1: 시작 알림
        yield json.dumps({"status": "progress", "message": "데이터 준비 중...", "percent": 10}) + "\n"
        time.sleep(0.5) 

        # Step 2: 훈련 시작 알림
        yield json.dumps({"status": "progress", "message": "AI 모델 훈련 중 (CPU 연산)...", "percent": 50}) + "\n"
        
        try:
            # Step 3: 실제 훈련 수행 (TrainML.py 실행)
            # 위에서 정의한 'csv_path_str' 변수를 사용합니다.
            results = run_training(sample_ids=sample_ids, dataset_path=csv_path_str)
            
            # Step 4: 완료 알림
            yield json.dumps({
                "status": "complete", 
                "message": "훈련 완료!", 
                "percent": 100,
                "results": results
            }) + "\n"
            
        except Exception as e:
            # 에러 발생 시 알림
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"

    # 스트리밍 응답 반환
    return StreamingHttpResponse(event_stream(), content_type="application/json")


# ============================================================
# 3. Test API
# ============================================================
def ping(request):
    return JsonResponse({"message": "pong"})