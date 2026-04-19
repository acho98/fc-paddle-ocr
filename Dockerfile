FROM paddlepaddle/paddle:3.3.1-gpu-cuda12.6-cudnn9.5-trt10.7

WORKDIR /app

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 모델 사전 다운로드 (빌드 시 1회, 런타임 다운로드 방지)
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
RUN python -c "\
from paddleocr import PaddleOCR; \
ocr = PaddleOCR(lang='korean', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False); \
print('모델 다운로드 완료')"

# 앱 복사
COPY main.py .

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
