#!/usr/bin/env bash
# Docling, Chandra(OCR), VLM, LLM 동작 및 GPU 사용 검증 스크립트
# 사용: 서버 실행 후 ./scripts/verify_all_services.sh [BASE_URL]
# 예: ./scripts/verify_all_services.sh http://127.0.0.1:8000

set -e
BASE_URL="${1:-http://127.0.0.1:8000}"

echo "=== Self-Hosting API 검증 (BASE_URL=$BASE_URL) ==="

echo ""
echo "1. Health & Models"
curl -s "$BASE_URL/health" | head -c 200 && echo ""
curl -s "$BASE_URL/v1/models" | head -c 300 && echo ""

echo ""
echo "2. LLM (vLLM, GPU) - chat completions"
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-llm-7b","messages":[{"role":"user","content":"Say hello in one word."}],"max_tokens":10}' \
  --max-time 120 | head -c 400
echo ""

echo ""
echo "3. VLM (GPU) - chat completions"
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-vlm-7b","messages":[{"role":"user","content":"What is 2+2? One number."}],"max_tokens":5}' \
  --max-time 120 | head -c 400
echo ""

echo ""
echo "4. OCR (Chandra, GPU) - 이미지 파일 필요"
if [ -f /tmp/ocr_test.png ]; then
  curl -s -X POST "$BASE_URL/v1/ocr" -F "image=@/tmp/ocr_test.png" -F "output_format=markdown" --max-time 180 | head -c 400
else
  echo "Skip (create /tmp/ocr_test.png for OCR test)"
fi
echo ""

echo ""
echo "5. Docling (GPU) - 문서 처리"
echo "Hello Docling verify." > /tmp/verify_docling.txt
curl -s -X POST "$BASE_URL/v1/documents/process" -F "file=@/tmp/verify_docling.txt" --max-time 60 | head -c 400
echo ""

echo ""
echo "6. GPU2 직렬화 (Docling → OCR 순차, 한 시점에 한 엔진만)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/verify_gpu2_serialization.sh" ]; then
  bash "$SCRIPT_DIR/verify_gpu2_serialization.sh" "$BASE_URL"
else
  echo "Docling then OCR 순서로 요청해 서버 로그에서 unload_other_gpu2_engines 확인"
fi
echo ""

echo "=== 검증 완료. 서버 로그에서 다음을 확인하세요: ==="
echo "  - vLLM: 'vLLM model ... loaded successfully', 'Generation completed: ... tokens/sec'"
echo "  - Docling: 'Docling GPU enabled: cuda:N'"
echo "  - Chandra: 'Chandra model initialized successfully'"
echo "  - GPU2 직렬화: Docling/OCR 처리 시 unload_other_gpu2_engines, 한 엔진만 로드"
