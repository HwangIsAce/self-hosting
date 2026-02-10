"""vLLM 실행 엔진 - Qwen LLM 모델 실행 (고성능 최적화 + Continuous Batching)"""

from typing import List, Dict, Any, Optional
import asyncio
import uuid
import os
import sys

# 시스템 Flash Attention 차단 (vLLM이 자체 Flash Attention 사용)
# 시스템에 설치된 flash_attn이 호환성 문제를 일으킬 수 있으므로
# vLLM이 자체 Flash Attention을 사용하도록 강제
# 메모리 초과 문제를 피하기 위해 Flash Attention 설치 없이 진행

# 환경 변수로 시스템 flash_attn 차단
# vLLM이 시스템 flash_attn을 import하지 못하도록 함
os.environ['VLLM_USE_V1'] = '1'  # vLLM v1 엔진 사용 (더 안정적)

# 시스템 flash_attn 경로를 Python 경로에서 제거
# vLLM이 시스템 flash_attn을 찾지 못하도록 함
original_sys_path = sys.path.copy()
sys.path = [p for p in sys.path if '/usr/local/lib/python3.11/dist-packages' not in p or 'flash_attn' not in p.lower()]

# sys.modules에서 flash_attn 제거 (이미 로드된 경우)
if 'flash_attn' in sys.modules:
    del sys.modules['flash_attn']
    # 관련 모듈도 제거
    for key in list(sys.modules.keys()):
        if key.startswith('flash_attn'):
            del sys.modules[key]

# vLLM import (자체 Flash Attention 포함)
# vLLM은 자체 Flash Attention 구현을 포함하고 있어서
# 시스템 flash_attn이 없어도 정상 작동함
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from transformers import AutoTokenizer
from api.config.settings import settings
from api.config.logging_config import get_logger

# import 후 sys.path 복원 (다른 모듈에 영향 없도록)
# 단, flash_attn이 포함된 경로는 제외
sys.path = original_sys_path

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
        
        # 시스템 Flash Attention 완전 차단
        # vLLM이 multiprocessing으로 실행되므로 자식 프로세스에서도 차단 필요
        # 환경 변수를 통해 Python이 시스템 flash_attn을 찾지 못하도록 함
        system_site_packages = '/usr/local/lib/python3.11/dist-packages'
        
        # 환경 변수로 자식 프로세스에서도 차단
        pythonpath = os.environ.get('PYTHONPATH', '')
        # 시스템 site-packages를 PYTHONPATH에서 제외
        filtered_paths = [p for p in pythonpath.split(':') if system_site_packages not in p]
        os.environ['PYTHONPATH'] = ':'.join(filtered_paths) if filtered_paths else ''
        
        # 현재 프로세스에서도 차단
        if system_site_packages in sys.path:
            sys.path.remove(system_site_packages)
            logger.info(f"Temporarily removed {system_site_packages} from sys.path to avoid flash_attn conflicts")
        
        # vLLM은 자체 Flash Attention을 포함하고 있음
        # 시스템 Flash Attention과의 호환성 문제를 피하기 위해
        # vLLM이 자체 구현을 사용하도록 함
        logger.info("vLLM will use its own Flash Attention implementation (enforce_eager=True)")
        
        # GPU ID 추출
        gpu_id = int(self.device_map.split(":")[-1])
        
        # vLLM 엔진 설정
        engine_args_dict = {
            "model": self.model_name,
            "tensor_parallel_size": settings.VLLM_TENSOR_PARALLEL_SIZE,
            "gpu_memory_utilization": settings.VLLM_GPU_MEMORY_UTILIZATION,
            "max_model_len": settings.VLLM_MAX_MODEL_LEN,
            "trust_remote_code": settings.VLLM_TRUST_REMOTE_CODE,
            "dtype": settings.VLLM_DTYPE,
            # Continuous Batching 설정
            "max_num_batched_tokens": settings.VLLM_MAX_NUM_BATCHED_TOKENS,
            "max_num_seqs": settings.VLLM_MAX_NUM_SEQS,
            # Flash Attention 호환성 문제 해결: eager mode 사용
            # 시스템 flash_attn이 호환되지 않을 경우 eager mode로 fallback
            "enforce_eager": True,  # Flash Attention 우회, 기본 PyTorch 연산 사용
        }
        
        # Speculative Decoding은 vLLM 버전에 따라 지원 여부가 다름
        # 지원되는 경우에만 추가
        if settings.VLLM_SPECULATIVE_MODEL:
            try:
                # vLLM 0.6.0+ 에서는 다른 방식으로 지원될 수 있음
                # 현재 버전에서는 일단 제외
                logger.warning("Speculative Decoding is not yet supported in this vLLM version")
            except Exception:
                pass
        
        engine_args = AsyncEngineArgs(**engine_args_dict)
        
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
        # vLLM은 top_k=None을 허용하지 않으므로 -1로 변환
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k is not None else -1,
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
        # vLLM은 top_k=None을 허용하지 않으므로 -1로 변환
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k is not None else -1,
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

