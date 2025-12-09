import json
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
# 1) class_id 목록 제공
# ============================================================
def get_classes(request):
    """
    backend/ML/ransomwaredataset.csv 파일을 읽어서 class_id 목록 반환
    """
    dataset_path = settings.BASE_DIR / "backend" / "ML" / "ransomwaredataset.csv"

    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        return JsonResponse({"error": f"Dataset load failed: {str(e)}"}, status=500)

    class_ids = sorted(df["class_id"].unique().tolist())

    return JsonResponse({"class_ids": class_ids})


# ============================================================
# 2. StreamingResponse로 훈련 과정 실시간 로그 제공
# ============================================================
@csrf_exempt
def train_stream(request):
    """
    실시간 학습 로그 스트리밍.
    프런트엔드 JS EventSource(SSE)로 수신 가능.
    """

    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        body = json.loads(request.body)
        class_id = body.get("class_id", None)
    except:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    if class_id is None:
        return JsonResponse({"error": "class_id is required"}, status=400)

    dataset_path = settings.BASE_DIR / "backend" / "ML"/ "ransomwaredataset.csv"

    def generate():
        """
        스트리밍 방식으로 로그 전송
        """
        yield "data: 학습 시작...\n\n"

        try:
            results = run_training(
                class_id=class_id,
                dataset_path=str(dataset_path),
                stream_callback=lambda msg: f"data: {msg}\n\n"
            )

            # 최종 결과 JSON 통째로 반환
            final_json = json.dumps({
                "status": "done",
                "class_id": class_id,
                "results": results
            })

            yield f"data: {final_json}\n\n"

        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingHttpResponse(
        generate(),
        content_type="text/event-stream"
    )


# ============================================================
# 3. 모델 훈련 API
# ============================================================
@csrf_exempt
def train_models(request):
    """
    POST 요청:
    {
        "class_ids": [1, 3, 5]
    }
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    # JSON 파싱
    try:
        body = json.loads(request.body)
        class_ids = body.get("class_ids", [])
    except:
        return JsonResponse({"error": "Invalid JSON"}, status=400)
    
    # 리스트가 비어있는지 확인
    if not class_ids or not isinstance(class_ids, list):
        return JsonResponse({"error": "class_ids list is required"}, status=400)

    # class_ids 필수
    if class_ids is None:
        return JsonResponse({"error": "class_id is required"}, status=400)

    dataset_path = settings.BASE_DIR / "backend" / "ML" / "ransomwaredataset.csv"

    try:
        # ML 훈련 코드 실행
        results = run_training(class_ids=class_ids, dataset_path=str(dataset_path))

        return JsonResponse({
            "message": "Training completed",
            "class_id": class_ids,
            "results": results
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


# ============================================================
# 3) Test API
# ============================================================
def ping(request):
    return JsonResponse({"message": "pong"})