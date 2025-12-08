import json
import pandas as pd
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from backend.ML.TrainML import run_training


# ============================================================
# 1) class_id 목록 제공
# ============================================================
def get_classes(request):
    """
    dataset.csv 파일을 읽어서 class_id 목록 반환
    """
    dataset_path = settings.BASE_DIR / "dataset.csv"

    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        return JsonResponse({"error": f"Dataset load failed: {str(e)}"}, status=500)

    class_ids = sorted(df["class_id"].unique().tolist())

    return JsonResponse({"class_ids": class_ids})


# ============================================================
# 2) 모델 훈련 API
# ============================================================
@csrf_exempt
def train_models(request):
    """
    POST 요청:
    {
        "class_id": 2
    }
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    # JSON 파싱
    try:
        body = json.loads(request.body)
        class_id = body.get("class_id", None)
    except:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    # class_id 필수
    if class_id is None:
        return JsonResponse({"error": "class_id is required"}, status=400)

    dataset_path = settings.BASE_DIR / "dataset.csv"

    try:
        # → 네가 만든 ML 훈련 코드 실행
        results = run_training(class_id=class_id, dataset_path=str(dataset_path))

        return JsonResponse({
            "message": "Training completed",
            "class_id": class_id,
            "results": results
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# ============================================================
# 3) Test API
# ============================================================
def ping(request):
    return JsonResponse({"message": "pong"})