#!/usr/bin/env bash
# 6단계: GPU2 직렬화 검증 — Docling → OCR 순서로 요청해 한 시점에 한 엔진만 로드되는지 확인.
# 사용: 서버 실행 후 ./scripts/verify_gpu2_serialization.sh [BASE_URL]
# ColPali는 HTTP 엔드포인트가 없으므로, 동일 직렬화 경로는 scripts/verify_colpali.py 로 별도 확인.

set -e
BASE_URL="${1:-http://127.0.0.1:8000}"

echo "=== 6단계: GPU2 직렬화 검증 (Docling → OCR) ==="
echo "BASE_URL=$BASE_URL"
echo ""

echo "6-1. Docling (GPU2) — 문서 처리"
echo "Hello Docling GPU2 serialization check." > /tmp/verify_gpu2_docling.txt
curl -s -X POST "$BASE_URL/v1/documents/process" -F "file=@/tmp/verify_gpu2_docling.txt" --max-time 120 | head -c 500
echo ""
echo ""

sleep 2
echo "6-2. OCR (Chandra, GPU2) — 이미지 처리"
if [ -f /tmp/ocr_test.png ]; then
  curl -s -X POST "$BASE_URL/v1/ocr" -F "image=@/tmp/ocr_test.png" -F "output_format=markdown" --max-time 180 | head -c 500
else
  echo "Skip (create /tmp/ocr_test.png for full OCR test)"
fi
echo ""
echo ""

echo "=== 6단계 검증 완료 ==="
echo "서버 로그에서 확인할 내용:"
echo "  - Docling 처리 시: unload_other_gpu2_engines(except_name=\"docling\"), 이후 Docling만 GPU2 사용"
echo "  - OCR 처리 시: unload_other_gpu2_engines(except_name=\"ocr\"), 이후 OCR만 GPU2 사용"
echo "  - 한 시점에 한 엔진만 로드되어 OOM 없이 순차 처리되는지 확인"
echo "ColPali 직렬화 검증: python scripts/verify_colpali.py (엔진 직접 호출)"
