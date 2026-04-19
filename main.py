"""
PaddleOCR GPU 서빙 서버 (멀티 GPU 지원)
POST /ocr/extract — 이미지 base64 리스트 → OCR 결과 (GPU 병렬 처리)
GET  /health     — 헬스체크
"""
import time, base64, re, os
from concurrent.futures import ThreadPoolExecutor, as_completed
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

from fastapi import FastAPI
from pydantic import BaseModel
from paddleocr import PaddleOCR
import numpy as np
from PIL import Image

app = FastAPI(title="PaddleOCR GPU Server")

# GPU 수 감지
GPU_COUNT = int(os.getenv("GPU_COUNT", "0"))
if GPU_COUNT == 0:
    import subprocess
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        GPU_COUNT = len([l for l in out.strip().split("\n") if l.startswith("GPU")])
    except Exception:
        GPU_COUNT = 1

print(f"감지된 GPU 수: {GPU_COUNT}")

# GPU별 OCR 인스턴스 생성
ocr_instances = []
for gpu_id in range(GPU_COUNT):
    print(f"GPU {gpu_id} 모델 로딩 중...")
    ocr = PaddleOCR(
        lang="korean",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=f"gpu:{gpu_id}",
    )
    ocr_instances.append(ocr)

# 워밍업 (각 GPU)
dummy = np.zeros((100, 100, 3), dtype=np.uint8)
Image.fromarray(dummy).save("/tmp/_warmup.jpg")
for gpu_id, ocr in enumerate(ocr_instances):
    ocr.predict("/tmp/_warmup.jpg")
    print(f"GPU {gpu_id} 워밍업 완료")
os.unlink("/tmp/_warmup.jpg")
print(f"모델 로딩 + 워밍업 완료 (GPU {GPU_COUNT}개)")

REG_NO = re.compile(r"\d{6}[-\u2013\u2014]?\d{7}")


class OCRRequest(BaseModel):
    images: list[str]  # base64 인코딩된 이미지 리스트
    detect_pii: bool = True


def _process_chunk(ocr_inst, img_b64_list, start_idx):
    """GPU 1개로 할당된 이미지 청크 처리"""
    tmp_paths = []
    for i, img_b64 in enumerate(img_b64_list):
        img_bytes = base64.b64decode(img_b64)
        path = f"/tmp/ocr_gpu_p{start_idx + i:02d}.jpg"
        with open(path, "wb") as f:
            f.write(img_bytes)
        tmp_paths.append(path)

    results = ocr_inst.predict(tmp_paths)

    pages = []
    for i, result in enumerate(results):
        page_num = start_idx + i + 1
        res = result.json["res"]
        texts = res.get("rec_texts", [])
        boxes = res.get("rec_boxes", [])
        pages.append({
            "page_num": page_num,
            "texts": texts,
            "boxes": boxes,
        })

    # 임시 파일 정리
    for p in tmp_paths:
        try:
            os.unlink(p)
        except OSError:
            pass

    return pages


@app.post("/ocr/extract")
def extract(req: OCRRequest):
    t0 = time.time()
    total = len(req.images)
    n_gpu = len(ocr_instances)

    # GPU별 이미지 분배
    chunk_size = (total + n_gpu - 1) // n_gpu
    chunks = []
    for g in range(n_gpu):
        start = g * chunk_size
        end = min(start + chunk_size, total)
        if start < total:
            chunks.append((g, start, req.images[start:end]))

    # 병렬 처리
    all_pages_raw = []
    if n_gpu == 1:
        all_pages_raw = _process_chunk(ocr_instances[0], req.images, 0)
    else:
        with ThreadPoolExecutor(max_workers=n_gpu) as pool:
            futures = {
                pool.submit(_process_chunk, ocr_instances[g], imgs, start): g
                for g, start, imgs in chunks
            }
            for fut in as_completed(futures):
                all_pages_raw.extend(fut.result())

    # 페이지 번호 순 정렬
    all_pages_raw.sort(key=lambda p: p["page_num"])

    # 결과 조립
    pages = []
    full_text_parts = []
    pages_with_regno = 0

    for raw in all_pages_raw:
        page_num = raw["page_num"]
        texts = raw["texts"]
        boxes = raw["boxes"]
        page_text = "\n".join(texts)

        masking_targets = []
        if req.detect_pii:
            for i, txt in enumerate(texts):
                if REG_NO.search(txt) and i < len(boxes):
                    masking_targets.append({
                        "full_text": txt,
                        "bbox": [int(x) for x in boxes[i]],
                    })
            if masking_targets:
                pages_with_regno += 1

        pages.append({
            "page": page_num,
            "text": page_text,
            "masking_targets": masking_targets,
        })
        full_text_parts.append(f"=== 페이지 {page_num} ===\n{page_text}")

    ocr_time = time.time() - t0
    return {
        "pages": pages,
        "full_text": "\n\n".join(full_text_parts),
        "stats": {
            "total_pages": total,
            "pages_with_regno": pages_with_regno,
            "ocr_time_sec": round(ocr_time, 2),
            "gpu_count": n_gpu,
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "engine": "PaddleOCR PP-OCRv5 (GPU, korean)",
        "gpu_count": len(ocr_instances),
        "lang": "korean",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
