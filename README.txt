============================================================
  PaddleOCR GPU 서빙 서버 — 설치 및 실행 가이드
============================================================

1. 사전 요구사항
------------------------------------------------------------
- Python 3.11
- NVIDIA GPU (A30/A100 등) + CUDA 12.x + cuDNN 설치 확인
- nvidia-smi 명령으로 GPU 인식 확인

  $ nvidia-smi -L
  GPU 0: NVIDIA A30 (...)
  GPU 1: NVIDIA A30 (...)


2. 의존성 설치
------------------------------------------------------------
# paddlepaddle-gpu (CUDA 12.6 기준, 다른 CUDA 버전이면 URL 변경)
$ pip install paddlepaddle-gpu -f https://www.paddlepaddle.org.cn/packages/stable/cu126/

# 나머지 패키지
$ pip install -r requirements.txt

# 설치 확인
$ python -c "import paddle; paddle.utils.run_check()"


3. 서버 실행
------------------------------------------------------------
# GPU 자동 감지 (nvidia-smi로 GPU 수 자동 인식)
$ nohup uvicorn main:app --host 0.0.0.0 --port 8002 > ocr_server.log 2>&1 &

# 또는 GPU 수 명시
$ GPU_COUNT=2 nohup uvicorn main:app --host 0.0.0.0 --port 8002 > ocr_server.log 2>&1 &

* 첫 실행 시 한국어 OCR 모델 자동 다운로드 (약 1~2분 소요)
* 이후 실행부터는 캐시 사용 (즉시 시작)


4. 동작 확인
------------------------------------------------------------
# 헬스체크
$ curl http://localhost:8002/health

# 정상 응답 예시:
# {"status":"ok","engine":"PaddleOCR PP-OCRv5 (GPU, korean)","gpu_count":2,"lang":"korean"}

# gpu_count가 2인지 확인!


5. API 사용법
------------------------------------------------------------
POST /ocr/extract
Content-Type: application/json

{
  "images": ["<base64 인코딩된 이미지>", ...],
  "detect_pii": true
}

응답:
{
  "pages": [{"page": 1, "text": "...", "masking_targets": [...]}],
  "full_text": "=== 페이지 1 ===\n...",
  "stats": {"total_pages": 12, "pages_with_regno": 2, "ocr_time_sec": 7.5, "gpu_count": 2}
}


6. 서버 중지
------------------------------------------------------------
$ pkill -f "uvicorn main:app"


7. 로그 확인
------------------------------------------------------------
$ tail -f ocr_server.log


8. 트러블슈팅
------------------------------------------------------------
Q: "No module named paddle" 에러
A: paddlepaddle-gpu 설치 확인. Python 3.11 필수.

Q: gpu_count가 1로 나옴
A: CUDA_VISIBLE_DEVICES 환경변수 확인. nvidia-smi로 GPU 2장 모두 보이는지 확인.

Q: 모델 다운로드 실패
A: 네트워크 확인. 또는 프록시 환경이면 HTTP_PROXY 설정.

Q: GPU 메모리 부족 (OOM)
A: PaddleOCR Mobile 모델은 ~20MB로 6GB 이상이면 충분. 다른 프로세스가 GPU 점유 중인지 확인.
