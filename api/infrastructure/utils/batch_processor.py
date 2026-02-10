"""배치 처리 유틸리티 - Chunk enrich 등에 사용"""

from typing import List, Dict, Any, Callable, Awaitable, Optional
import asyncio
from api.config.logging_config import get_logger

logger = get_logger(__name__)


async def process_batch(
    items: List[Any],
    processor: Callable[[Any], Awaitable[Any]],
    batch_size: int = 5,
    max_concurrent: Optional[int] = None
) -> List[Any]:
    """
    아이템 리스트를 배치로 처리
    
    Args:
        items: 처리할 아이템 리스트
        processor: 각 아이템을 처리하는 async 함수
        batch_size: 배치 크기 (vLLM이 자동 배치 처리하므로 큰 값도 가능)
        max_concurrent: 최대 동시 실행 수 (None = 제한 없음)
    
    Returns:
        처리 결과 리스트 (입력 순서와 동일)
    
    Example:
        # 100개 chunk enrich
        chunks = ["chunk1", "chunk2", ..., "chunk100"]
        results = await process_batch(
            chunks,
            lambda chunk: llm_engine.generate(
                messages=[{"role": "user", "content": f"Enrich: {chunk}"}]
            ),
            batch_size=10
        )
    """
    if max_concurrent:
        # Semaphore로 동시 실행 수 제한
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(item):
            async with semaphore:
                return await processor(item)
        
        processor = process_with_semaphore
    
    # 배치로 나누기
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(items)-1)//batch_size + 1} ({len(batch)} items)")
        
        # 배치 내 아이템들을 동시에 처리
        batch_results = await asyncio.gather(*[processor(item) for item in batch])
        results.extend(batch_results)
    
    return results


async def enrich_chunks_batch(
    chunks: List[str],
    llm_engine,
    prompt_template: str = "Enrich this chunk with additional context: {chunk}",
    batch_size: int = 10,
    max_tokens: int = 300,
    **generation_kwargs
) -> List[Dict[str, Any]]:
    """
    Chunk 리스트를 배치로 enrich
    
    Args:
        chunks: Enrich할 chunk 리스트
        llm_engine: LLM 엔진 (VLLMEngine 또는 LLMEngine)
        prompt_template: 프롬프트 템플릿
        batch_size: 배치 크기
        max_tokens: 최대 생성 토큰 수
        **generation_kwargs: 추가 생성 파라미터
    
    Returns:
        Enrich 결과 리스트
    """
    # vLLM 엔진인 경우 generate_batch 사용
    if hasattr(llm_engine, 'generate_batch'):
        logger.info(f"Using vLLM batch processing for {len(chunks)} chunks")
        
        # 모든 chunk를 메시지 리스트로 변환
        messages_list = [
            [{"role": "user", "content": prompt_template.format(chunk=chunk)}]
            for chunk in chunks
        ]
        
        # 배치로 처리
        results = await llm_engine.generate_batch(
            messages_list,
            max_tokens=max_tokens,
            **generation_kwargs
        )
        
        return results
    else:
        # HuggingFace 엔진인 경우 순차 처리
        logger.info(f"Using sequential processing for {len(chunks)} chunks")
        
        async def process_chunk(chunk):
            return await llm_engine.generate(
                messages=[{"role": "user", "content": prompt_template.format(chunk=chunk)}],
                max_tokens=max_tokens,
                **generation_kwargs
            )
        
        return await process_batch(chunks, process_chunk, batch_size=batch_size)

