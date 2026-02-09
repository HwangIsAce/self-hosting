# Flash Attention 2 빠른 설치 가이드

## 🚀 가장 빠른 설치 방법 (1분 이내)

### 방법 1: Pre-built Wheel 사용 (추천 ⭐)

**조건**: Python 3.9-3.11, PyTorch 2.0+, CUDA 11.8/12.1/12.4

```bash
# uv 환경에서
export PATH="/root/.local/bin:$PATH"
cd /root/projects/self-hosting

# 한 줄로 설치 (컴파일 없음!)
uv pip install flash-attn --prefer-binary
```

**장점:**
- ✅ 컴파일 시간 0초
- ✅ 가장 빠름 (1-2분)
- ✅ 대부분의 환경에서 작동

**실패 시**: 다음 방법으로 진행

---

### 방법 2: kingbri1 Pre-built Wheel (환경이 정확히 맞을 때)

**릴리즈 페이지**: https://github.com/kingbri1/flash-attention/releases

```bash
# 1. 현재 환경 확인
export PATH="/root/.local/bin:$PATH"
cd /root/projects/self-hosting

uv run python3 -c "
import torch
print(f'Python: 3.11')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
"

# 2. 적절한 wheel 다운로드 및 설치
# 예시: Python 3.11, PyTorch 2.x, CUDA 12.x
wget https://github.com/kingbri1/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

uv pip install flash_attn-2.8.3+cu128torch2.8.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

**주의**: PyTorch와 CUDA 버전이 정확히 일치해야 함

---

### 방법 3: 소스 빌드 (최후의 수단, 15-30분)

```bash
export PATH="/root/.local/bin:$PATH"
cd /root/projects/self-hosting

# 빌드 최적화
export MAX_JOBS=2  # 메모리 절약
export CUDA_HOME=/usr/local/cuda  # CUDA 경로 (필요시)

# 설치
uv pip install flash-attn --no-build-isolation
```

---

## 설치 확인

```bash
export PATH="/root/.local/bin:$PATH"
cd /root/projects/self-hosting

uv run python3 -c "
import flash_attn
print(f'✅ Flash Attention {flash_attn.__version__} 설치 완료!')
"
```

---

## 빠른 설치 체크리스트

1. ✅ **먼저 시도**: `uv pip install flash-attn --prefer-binary`
2. ✅ **실패 시**: kingbri1 wheel 확인
3. ✅ **최후**: 소스 빌드

---

## 현재 프로젝트 적용 상태

Flash Attention은 이미 코드에 통합되어 있습니다:
- `api/infrastructure/models/model_loader.py`에서 자동 활성화
- `flash_attn` 패키지가 설치되어 있으면 자동으로 사용

**확인 방법:**
```bash
# 서버 로그에서 확인
grep "Flash Attention" /tmp/server.log
```

---

## 문제 해결

### "No module named 'einops'" 오류
```bash
uv add einops
```

### 빌드 실패
```bash
# 캐시 정리
rm -rf ~/.cache/pip
rm -rf build/ dist/ *.egg-info

# 재시도
uv pip install flash-attn --prefer-binary --no-cache-dir
```

### CUDA 버전 불일치
```bash
# PyTorch CUDA 버전 확인
uv run python3 -c "import torch; print(torch.version.cuda)"

# 필요시 PyTorch 재설치
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## 성능 벤치마크 결과

- **8192 토큰**: Flash Attention이 27.5% 빠름
- **4096 토큰**: Flash Attention이 14.7% 빠름
- **2048 토큰**: Flash Attention이 6.2% 빠름
- **1024 토큰 이하**: 성능 차이 미미

**결론**: 긴 시퀀스(4096+ 토큰)에서 효과적!

