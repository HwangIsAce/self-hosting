"""vLLM 실행 엔진 - Qwen LLM 모델 실행 (고성능 최적화 + Continuous Batching)"""

from typing import List, Dict, Any, Optional
import asyncio
import uuid
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer
from api.config.settings import settings
from api.config.logging_config import get_logger

logger = get_logger(__name__)


class VLLMEngine:
    """vLLM 기반 LLM 실행 엔진 (고성능 + Continuous Batching 지원)"""
    
    def __init__(
        self,
        model_name: str,
        device_map: Optional[str] = None,
        use_quantization: bool = False
    ):
        self.model_name = model_name
        self.device_map = device_map or f"cuda:{settings.LLM_GPU_ID}"
        self.use_quantization = use_quantization
        self.llm: Optional[AsyncLLMEngine] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self._loaded = False
        self._request_counter = 0  # 고유 request_id 생성용
    
    def load_model(self):
        """vLLM 모델 로드"""
        if self._loaded:
            return
        
        logger.info(f"Loading vLLM model: {self.model_name}")
        
        # GPU ID 추출
        gpu_id = int(self.device_map.split(":")[-1])
        
        # vLLM 엔진 설정
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            tensor_parallel_size=settings.VLLM_TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=settings.VLLM_GPU_MEMORY_UTILIZATION,
            max_model_len=settings.VLLM_MAX_MODEL_LEN,
            trust_remote_code=settings.VLLM_TRUST_REMOTE_CODE,
            dtype=settings.VLLM_DTYPE,
            # Continuous Batching 설정
            max_num_batched_tokens=settings.VLLM_MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=settings.VLLM_MAX_NUM_SEQS,
            # Speculative Decoding (선택사항)
            speculative_model=settings.VLLM_SPECULATIVE_MODEL if settings.VLLM_SPECULATIVE_MODEL else None,
            num_speculative_tokens=settings.VLLM_NUM_SPECULATIVE_TOKENS if settings.VLLM_SPECULATIVE_MODEL else 0,
        )
        
        # vLLM 엔진 초기화
        self.llm = AsyncLLMEngine.from_engine_args(engine_args)
        
        # 토크나이저 로드 (채팅 템플릿용)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=settings.VLLM_TRUST_REMOTE_CODE
        )
        
        self._loaded = True
        logger.info(f"vLLM model {self.model_name} loaded successfully")
        logger.info(f"  - GPU Memory Utilization: {settings.VLLM_GPU_MEMORY_UTILIZATION}")
        logger.info(f"  - Max Sequences: {settings.VLLM_MAX_NUM_SEQS}")
        if settings.VLLM_SPECULATIVE_MODEL:
            logger.info(f"  - Speculative Decoding: {settings.VLLM_SPECULATIVE_MODEL}")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """텍스트 생성 (vLLM - 단일 요청)"""
        if not self._loaded:
            self.load_model()
        
        # 메시지를 프롬프트로 변환
        prompt = self._format_messages(messages)
        
        # SamplingParams 설정
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=stop or [],
        )
        
        # 고유 request_id 생성
        self._request_counter += 1
        request_id = f"req_{self._request_counter}_{uuid.uuid4().hex[:8]}"
        
        # vLLM으로 생성 (비동기)
        final_output = None
        async for request_output in self.llm.generate(
            prompt,
            sampling_params,
            request_id
        ):
            final_output = request_output
        
        if final_output is None:
            raise RuntimeError("vLLM generation failed: no output received")
        
        # 최종 결과 추출
        output = final_output.outputs[0]
        generated_text = output.text
        
        # 토큰 사용량 계산
        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = len(output.token_ids)
        total_tokens = prompt_tokens + completion_tokens
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": output.finish_reason or "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }
    
    async def generate_batch(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 0.9,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        배치 텍스트 생성 (vLLM - 여러 요청 동시 처리)
        
        Continuous Batching을 활용하여 여러 요청을 효율적으로 처리합니다.
        예: 100개 chunk enrich 시 4-7배 빠름
        
        Args:
            messages_list: 메시지 리스트의 리스트 (각각이 하나의 요청)
            
        Returns:
            결과 리스트 (입력 순서와 동일)
        """
        if not self._loaded:
            self.load_model()
        
        # SamplingParams 설정
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            stop=stop or [],
        )
        
        # 모든 요청을 동시에 제출 (vLLM이 자동으로 배치 처리)
        tasks = []
        request_ids = []
        
        for i, messages in enumerate(messages_list):
            prompt = self._format_messages(messages)
            request_id = f"batch_{self._request_counter}_{i}_{uuid.uuid4().hex[:8]}"
            request_ids.append(request_id)
            
            # 각 요청을 비동기 태스크로 생성
            tasks.append(self._generate_single(prompt, sampling_params, request_id))
        
        self._request_counter += len(messages_list)
        
        # 모든 요청을 동시에 실행 (vLLM이 자동 배치 처리)
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def _generate_single(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str
    ) -> Dict[str, Any]:
        """단일 요청 생성 (내부 헬퍼)"""
        final_output = None
        async for request_output in self.llm.generate(
            prompt,
            sampling_params,
            request_id
        ):
            final_output = request_output
        
        if final_output is None:
            raise RuntimeError(f"vLLM generation failed for {request_id}")
        
        output = final_output.outputs[0]
        generated_text = output.text
        
        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = len(output.token_ids)
        total_tokens = prompt_tokens + completion_tokens
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": output.finish_reason or "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """메시지 리스트를 프롬프트로 변환"""
        # Qwen 모델의 채팅 템플릿 사용
        try:
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        except Exception as e:
            logger.warning(f"Failed to use chat template: {str(e)}, using simple formatting")
        
        # 간단한 포맷팅 (fallback)
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
        formatted += "Assistant: "
        return formatted

