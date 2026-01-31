# Self-Hosting API

OpenAI-compatible API Gateway for Qwen LLM, VLM, and Docling/Chandra services running on RunPod RTX 4090 GPUs.

## 📚 문서

- [배포 가이드 (Deployment Guide)](./DEPLOYMENT.md) - 상세한 배포 및 설정 방법

## 🏗️ 아키텍처

```
Client → API Gateway (FastAPI) → RunPod Pods
                                ├── RTX 4090 #1: Qwen LLM
                                ├── RTX 4090 #2: Qwen VLM
                                └── RTX 4090 #3: Docling/Chandra
```

## 🎯 사용 방법

### 빠른 시작

#### Docker Compose (권장)

```bash
# 환경 변수 설정 (.env 파일 생성)
cp .env.example .env
# .env 파일 편집: RunPod Pod 엔드포인트 입력

# 배포
docker-compose up -d

# 또는 배포 스크립트 사용
./scripts/deploy.sh
```

#### 로컬 실행

```bash
# 의존성 설치
uv sync

# 서버 시작
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 또는 스크립트 사용
./scripts/start.sh
```

#### API 문서

서버 실행 후 다음 URL에서 API 문서를 확인할 수 있습니다:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

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
