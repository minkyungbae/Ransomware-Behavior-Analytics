import json
import random
import pandas as pd
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.shortcuts import render

from ML.TrainML import run_training


def mainpage(request):
    """
    mainpage.html 렌더링
    """
    return render(request, "mainpage.html")


# ============================================================
# 1. sample_id 목록 제공 (수정됨)
# ============================================================
def get_samples(request):
    """
    backend/ML/ransomwaredataset.csv 파일을 읽어서 
    sample_id 중 랜덤으로 10개를 뽑아 반환
    """
    dataset_path = settings.BASE_DIR / "backend" / "ML" / "ransomwaredataset.csv"

    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        return JsonResponse({"error": f"Dataset load failed: {str(e)}"}, status=500)

    # 1. 전체 고유 ID 리스트 추출
    all_sample_ids = df["sample_id"].unique().tolist()
    
    # 2. 랜덤 10개 추출 로직
    target_count = 10
    
    if len(all_sample_ids) > target_count:
        # ID가 10개보다 많으면 랜덤으로 10개 비복원 추출
        sample_ids = random.sample(all_sample_ids, target_count)
    else:
        # ID가 10개 이하라면 전체 반환
        sample_ids = all_sample_ids

    # (선택 사항) UI 가독성을 위해 추출된 10개를 정렬해서 보냄
    sample_ids.sort()

    return JsonResponse({"sample_ids": sample_ids})


# ============================================================
# 2. 모델 훈련 API
# ============================================================
@csrf_exempt
def train_models(request):
    """
    POST 요청:
    {
        "sample_ids": [1, 3, 5]
    }
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    # JSON 파싱
    try:
        body = json.loads(request.body)
        sample_ids = body.get("sample_ids", [])
    except:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    # 리스트가 비어있는지 확인
    if not sample_ids or not isinstance(sample_ids, list):
        return JsonResponse({"error": "sample_ids list is required"}, status=400)

    # sample_ids 필수
    if sample_ids is None:
        return JsonResponse({"error": "sample_id is required"}, status=400)

    dataset_path = settings.BASE_DIR / "backend" / "ML" / "ransomwaredataset.csv"

    try:
        # ML 훈련 코드 실행
        results = run_training(sample_ids=sample_ids, dataset_path=str(dataset_path))

        return JsonResponse({
            "message": "Training completed",
            "class_id": sample_ids,
            "results": results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({"error": f"모델 훈련 중 오류 발생: {type(e).__name__} - {str(e)}"}, status=500)


# ============================================================
# 3. Test API
# ============================================================
def ping(request):
    return JsonResponse({"message": "pong"})