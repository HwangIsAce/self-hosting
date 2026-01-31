#!/usr/bin/env python3
"""RunPod Pod 헬스체크 스크립트"""

import asyncio
import sys
from api.config.settings import settings
from api.infrastructure.clients.runpod_client import RunPodClientFactory
from api.config.logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def check_pod_health(pod_name: str, endpoint: str):
    """Pod 헬스체크"""
    try:
        if pod_name == "llm":
            client = RunPodClientFactory.get_llm_client()
        elif pod_name == "vlm":
            client = RunPodClientFactory.get_vlm_client()
        elif pod_name == "docling":
            client = RunPodClientFactory.get_docling_client()
        else:
            logger.error(f"Unknown pod: {pod_name}")
            return False
        
        health = await client.health_check()
        status = health.get("status", "unknown")
        
        if status == "healthy":
            logger.info(f"✅ {pod_name.upper()} Pod ({endpoint}): {status}")
            return True
        else:
            logger.warning(f"⚠️  {pod_name.upper()} Pod ({endpoint}): {status}")
            if "error" in health:
                logger.warning(f"   Error: {health['error']}")
            return False
    except Exception as e:
        logger.error(f"❌ {pod_name.upper()} Pod ({endpoint}): Connection failed - {str(e)}")
        return False


async def main():
    """메인 함수"""
    logger.info("Checking RunPod Pod health...")
    logger.info("")
    
    results = []
    
    # LLM Pod 체크
    if settings.RUNPOD_LLM_ENDPOINT:
        results.append(await check_pod_health("llm", settings.RUNPOD_LLM_ENDPOINT))
    else:
        logger.warning("⚠️  LLM Pod endpoint not configured")
        results.append(False)
    
    # VLM Pod 체크
    if settings.RUNPOD_VLM_ENDPOINT:
        results.append(await check_pod_health("vlm", settings.RUNPOD_VLM_ENDPOINT))
    else:
        logger.warning("⚠️  VLM Pod endpoint not configured")
        results.append(False)
    
    # Docling Pod 체크
    if settings.RUNPOD_DOCLING_ENDPOINT:
        results.append(await check_pod_health("docling", settings.RUNPOD_DOCLING_ENDPOINT))
    else:
        logger.warning("⚠️  Docling Pod endpoint not configured")
        results.append(False)
    
    logger.info("")
    
    # 결과 요약
    all_healthy = all(results)
    if all_healthy:
        logger.info("✅ All pods are healthy!")
        return 0
    else:
        logger.error("❌ Some pods are unhealthy or not configured")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
