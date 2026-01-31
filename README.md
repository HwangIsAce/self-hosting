# Self-Hosting API

OpenAI-compatible API Gateway for Qwen LLM, VLM, and Docling/Chandra services running on RunPod RTX 4090 GPUs.

## 🏗️ 아키텍처

```
Client → API Gateway (FastAPI) → RunPod Pods
                                ├── RTX 4090 #1: Qwen LLM
                                ├── RTX 4090 #2: Qwen VLM
                                └── RTX 4090 #3: Docling/Chandra
```

## 🎯 사용 방법

### 서버 시작

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### API 사용 예시

#### 1. 모델 목록 조회

```bash
curl http://localhost:8000/v1/models
```

#### 2. Chat Completions (LLM)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-llm-7b",
    "messages": [
      {"role": "user", "content": "안녕하세요!"}
    ],
    "temperature": 0.7,
    "max_tokens": 1000
  }'
```

#### 3. Chat Completions (VLM - 이미지 포함)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen-vlm-7b",
    "messages": [
      {
        "role": "user",
        "content": "이 이미지에 무엇이 보이나요?",
        "image_url": "https://example.com/image.jpg"
      }
    ],
    "max_tokens": 500
  }'
```

#### 4. 문서 처리 (Docling)

```bash
curl -X POST http://localhost:8000/v1/documents/process \
  -F "file=@document.pdf"
```

### Python 클라이언트 사용

```python
import httpx

async def main():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "qwen-llm-7b",
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ]
            }
        )
        print(response.json())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```
