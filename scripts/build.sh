#!/bin/bash

# Docker 이미지 빌드 스크립트

set -e

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Building Docker image for Self-Hosting API...${NC}"

# 이미지 태그
IMAGE_NAME="self-hosting-api"
IMAGE_TAG="${1:-latest}"

# 빌드
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

echo -e "${GREEN}Build complete!${NC}"
echo -e "${GREEN}Image: ${IMAGE_NAME}:${IMAGE_TAG}${NC}"
echo ""
echo -e "${BLUE}To run the container:${NC}"
echo "docker run -d -p 8000:8000 --env-file .env ${IMAGE_NAME}:${IMAGE_TAG}"
