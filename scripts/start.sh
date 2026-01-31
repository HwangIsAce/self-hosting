#!/bin/bash

# Self-Hosting API 서버 시작 스크립트

set -e

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Self-Hosting API Gateway...${NC}"

# .env 파일 확인
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found. Using .env.example as template.${NC}"
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${YELLOW}Please edit .env file with your RunPod Pod endpoints.${NC}"
    else
        echo -e "${RED}Error: .env.example file not found.${NC}"
        exit 1
    fi
fi

# Python 버전 확인
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}Python version: ${python_version}${NC}"

# 의존성 확인
if ! command -v uv &> /dev/null; then
    echo -e "${YELLOW}uv not found. Installing dependencies with pip...${NC}"
    pip install -e .
else
    echo -e "${GREEN}Using uv for dependency management${NC}"
    uv sync
fi

# 서버 시작
echo -e "${GREEN}Starting server...${NC}"
echo -e "${GREEN}API will be available at: http://localhost:8000${NC}"
echo -e "${GREEN}API docs will be available at: http://localhost:8000/docs${NC}"
echo ""

if command -v uv &> /dev/null; then
    uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
else
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
fi
