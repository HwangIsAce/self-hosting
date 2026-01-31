# 배포 가이드 (Deployment Guide)

Self-Hosting API Gateway를 배포하는 방법을 안내합니다.

## 📋 사전 요구사항

### 필수 요구사항

1. **RunPod Pods**: 다음 3개의 RunPod Pod가 실행 중이어야 합니다
   - **Qwen LLM Pod** (RTX 4090): Hugging Face에서 Qwen LLM 모델 다운로드하여 사용
   - **Qwen VLM Pod** (RTX 4090): Hugging Face에서 Qwen VLM 모델 다운로드하여 사용
   - **Docling Pod** (RTX 4090): Docling 프레임워크 배포 (필요시 Chandra OCR 모델 사용)
     - Docling: [문서 처리 프레임워크](https://docling-project.github.io/docling/)
     - Chandra: [Hugging Face OCR 모델](https://huggingface.co/datalab-to/chandra)

2. **RunPod Pod 엔드포인트**: 각 Pod의 고유 URL
   - 예시: `https://xxx-llm-1234.runpod.net`

### 선택 요구사항

- **Docker** (Docker 배포 시)
- **Python 3.12+** (로컬 배포 시)
- **uv** (권장 패키지 관리자)

## 🔧 환경 설정

### 1. 환경 변수 파일 생성

프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 입력하세요:

```bash
# RunPod Pod 엔드포인트
# Qwen LLM: Hugging Face 모델을 다운로드하여 Pod에 배포
RUNPOD_LLM_ENDPOINT=https://your-llm-pod.runpod.net

# Qwen VLM: Hugging Face 모델을 다운로드하여 Pod에 배포
RUNPOD_VLM_ENDPOINT=https://your-vlm-pod.runpod.net

# Docling: Docling 프레임워크를 Pod에 배포 (필요시 Chandra OCR 모델 사용)
RUNPOD_DOCLING_ENDPOINT=https://your-docling-pod.runpod.net

# RunPod API 키 (선택사항)
RUNPOD_API_KEY=your-api-key-here

# 서버 설정
HOST=0.0.0.0
PORT=8000
DEBUG=false

# HTTP 클라이언트 설정
REQUEST_TIMEOUT=300
MAX_RETRIES=3
MAX_CONNECTIONS=100

# CORS 설정
CORS_ORIGINS=*

# 로깅
LOG_LEVEL=INFO
```

### 2. RunPod Pod 엔드포인트 확인

RunPod 대시보드에서 각 Pod의 엔드포인트를 확인하고 `.env` 파일에 입력하세요.

## 🚀 배포 방법

### 방법 1: Docker Compose (권장)

가장 간단한 배포 방법입니다.

```bash
# 배포 스크립트 사용
./scripts/deploy.sh
# 또는 직접 실행
docker-compose up -d --build
```

**명령어:**
- 시작: `docker-compose up -d`
- 중지: `docker-compose down`
- 로그 확인: `docker-compose logs -f`
- 재시작: `docker-compose restart`

### 방법 2: Docker (Standalone)

```bash
# 이미지 빌드
./scripts/build.sh
# 또는
docker build -t self-hosting-api:latest .

# 컨테이너 실행
docker run -d \
  --name self-hosting-api \
  -p 8000:8000 \
  --env-file .env \
  --restart unless-stopped \
  self-hosting-api:latest
```

**명령어:**
- 로그 확인: `docker logs -f self-hosting-api`
- 중지: `docker stop self-hosting-api`
- 삭제: `docker rm self-hosting-api`

### 방법 3: 로컬 배포

```bash
# 의존성 설치
uv sync
# 또는
pip install -e .

# 서버 시작
uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
# 또는
./scripts/start.sh
```

## ✅ 배포 확인

### 1. 헬스체크

```bash
curl http://localhost:8000/health
```

예상 응답:
```json
{"status": "healthy"}
```

### 2. API 문서 확인

브라우저에서 다음 URL을 열어보세요:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 3. 모델 목록 확인

```bash
curl http://localhost:8000/v1/models
```

### 4. RunPod Pod 연결 확인

```bash
python scripts/health_check.py
```

## 🔍 문제 해결

### 포트가 이미 사용 중인 경우

다른 포트를 사용하거나 기존 프로세스를 종료하세요:

```bash
# 포트 8000 사용 중인 프로세스 확인
lsof -i :8000

# 프로세스 종료
kill -9 <PID>
```

### RunPod Pod 연결 실패

1. `.env` 파일의 엔드포인트가 올바른지 확인
2. RunPod Pod가 실행 중인지 확인
3. 네트워크 연결 확인

```bash
# Pod 엔드포인트 테스트
curl https://your-llm-pod.runpod.net/health
```

### Docker 빌드 실패

1. Docker가 실행 중인지 확인
2. Dockerfile의 경로가 올바른지 확인
3. 빌드 로그 확인: `docker build --no-cache -t self-hosting-api .`

## 📊 모니터링

### 로그 확인

**Docker Compose:**
```bash
docker-compose logs -f api-gateway
```

**Docker:**
```bash
docker logs -f self-hosting-api
```

**로컬:**
로그는 콘솔에 출력됩니다.

### 메트릭 (향후 추가 예정)

- 요청 수
- 응답 시간
- 에러율
- RunPod Pod 상태

## 🔒 보안 고려사항

1. **환경 변수 보호**: `.env` 파일을 Git에 커밋하지 마세요
2. **API 키 관리**: RunPod API 키를 안전하게 관리하세요
3. **CORS 설정**: 프로덕션에서는 특정 도메인만 허용하세요
4. **HTTPS**: 프로덕션에서는 역방향 프록시(Nginx, Traefik)를 사용하여 HTTPS를 적용하세요

## 🌐 프로덕션 배포

### Nginx 역방향 프록시 설정 예시

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### SSL/TLS 인증서

Let's Encrypt를 사용하여 무료 SSL 인증서를 발급받을 수 있습니다:

```bash
sudo certbot --nginx -d api.yourdomain.com
```

## 📝 추가 리소스

- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [RunPod 문서](https://docs.runpod.io/)
- [Docker 문서](https://docs.docker.com/)
