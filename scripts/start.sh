#!/bin/bash
# Self-Hosting API 서버 시작 스크립트
# 실행 시 GPU 0,1,2 사용 중인 프로세스를 먼저 종료한 뒤 서버를 띄웁니다.

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting Self-Hosting API Gateway...${NC}"

# --- GPU 0,1,2 기존 캐시(프로세스) 정리 ---
if command -v nvidia-smi &>/dev/null; then
  echo -e "${YELLOW}Clearing GPU 0,1,2 (killing existing compute processes)...${NC}"
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u) || true
  if [ -n "$pids" ]; then
    for pid in $pids; do
      [ -z "$pid" ] && continue
      if kill -0 "$pid" 2>/dev/null; then
        echo "  Killing PID $pid"
        kill -TERM "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
      fi
    done
    sleep 2
    for pid in $pids; do
      [ -z "$pid" ] && continue
      if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
      fi
    done
    sleep 1
    echo -e "${GREEN}GPU processes cleared.${NC}"
  else
    echo -e "${GREEN}No GPU compute processes to clear.${NC}"
  fi
else
  echo -e "${YELLOW}nvidia-smi not found, skipping GPU clear.${NC}"
fi

# .env 파일 확인
if [ ! -f .env ]; then
  echo -e "${YELLOW}Warning: .env file not found.${NC}"
  if [ -f .env.example ]; then
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env with your settings.${NC}"
  fi
fi

# 의존성
if command -v uv &>/dev/null; then
  echo -e "${GREEN}Using uv${NC}"
  uv sync
else
  echo -e "${YELLOW}uv not found. Install dependencies with: pip install -e .${NC}"
fi

# 서버 시작
echo -e "${GREEN}Starting server...${NC}"
echo -e "${GREEN}API: http://0.0.0.0:8000  (docs: http://0.0.0.0:8000/docs)${NC}"
echo ""

if command -v uv &>/dev/null; then
  exec uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
else
  exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
fi
