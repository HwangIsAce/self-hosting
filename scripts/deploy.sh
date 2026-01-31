#!/bin/bash

# Self-Hosting API 배포 스크립트

set -e

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Self-Hosting API Deployment Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# .env 파일 확인
if [ ! -f .env ]; then
    echo -e "${YELLOW}Warning: .env file not found.${NC}"
    echo -e "${YELLOW}Please create .env file with your RunPod Pod endpoints.${NC}"
    echo -e "${YELLOW}You can use the following template:${NC}"
    echo ""
    echo "RUNPOD_LLM_ENDPOINT=https://your-llm-pod.runpod.net"
    echo "RUNPOD_VLM_ENDPOINT=https://your-vlm-pod.runpod.net"
    echo "RUNPOD_DOCLING_ENDPOINT=https://your-docling-pod.runpod.net"
    echo "RUNPOD_API_KEY=your-api-key-here"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 배포 방식 선택
echo -e "${GREEN}Select deployment method:${NC}"
echo "1) Docker Compose (Recommended)"
echo "2) Docker (Standalone)"
echo "3) Local (uv/uvicorn)"
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo -e "${GREEN}Deploying with Docker Compose...${NC}"
        if ! command -v docker-compose &> /dev/null && ! command -v docker &> /dev/null; then
            echo -e "${RED}Error: Docker is not installed.${NC}"
            exit 1
        fi
        
        # Docker Compose로 빌드 및 실행
        if command -v docker-compose &> /dev/null; then
            docker-compose up -d --build
        else
            docker compose up -d --build
        fi
        
        echo -e "${GREEN}Deployment complete!${NC}"
        echo -e "${GREEN}API available at: http://localhost:8000${NC}"
        echo -e "${GREEN}API docs at: http://localhost:8000/docs${NC}"
        echo ""
        echo -e "${YELLOW}To view logs: docker-compose logs -f${NC}"
        echo -e "${YELLOW}To stop: docker-compose down${NC}"
        ;;
    2)
        echo -e "${GREEN}Deploying with Docker...${NC}"
        if ! command -v docker &> /dev/null; then
            echo -e "${RED}Error: Docker is not installed.${NC}"
            exit 1
        fi
        
        # Docker 이미지 빌드
        echo -e "${BLUE}Building Docker image...${NC}"
        docker build -t self-hosting-api:latest .
        
        # 컨테이너 실행
        echo -e "${BLUE}Starting container...${NC}"
        docker run -d \
            --name self-hosting-api \
            -p 8000:8000 \
            --env-file .env \
            --restart unless-stopped \
            self-hosting-api:latest
        
        echo -e "${GREEN}Deployment complete!${NC}"
        echo -e "${GREEN}API available at: http://localhost:8000${NC}"
        echo -e "${GREEN}API docs at: http://localhost:8000/docs${NC}"
        echo ""
        echo -e "${YELLOW}To view logs: docker logs -f self-hosting-api${NC}"
        echo -e "${YELLOW}To stop: docker stop self-hosting-api && docker rm self-hosting-api${NC}"
        ;;
    3)
        echo -e "${GREEN}Deploying locally...${NC}"
        
        # Python 버전 확인
        python_version=$(python3 --version 2>&1 | awk '{print $2}')
        echo -e "${GREEN}Python version: ${python_version}${NC}"
        
        # 의존성 설치
        if command -v uv &> /dev/null; then
            echo -e "${BLUE}Installing dependencies with uv...${NC}"
            uv sync
            echo -e "${BLUE}Starting server with uv...${NC}"
            uv run uvicorn api.main:app --host 0.0.0.0 --port 8000
        else
            echo -e "${YELLOW}uv not found. Using pip...${NC}"
            pip install -e .
            echo -e "${BLUE}Starting server...${NC}"
            uvicorn api.main:app --host 0.0.0.0 --port 8000
        fi
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac
