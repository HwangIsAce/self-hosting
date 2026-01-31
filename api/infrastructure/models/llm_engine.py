"""LLM 실행 엔진 - Qwen LLM 모델 실행"""

from typing import List, Dict, Any, Optional
import torch
from transformers import TextIteratorStreamer
import asyncio
from threading import Thread

from api.infrastructure.models.model_loader import ModelLoader
from api.config.logging_config import get_logger

logger = get_logger(__name__)


class LLMEngine:
    """Qwen LLM 실행 엔진"""
    
    def __init__(
        self,
        model_name: str,
        device_map: Optional[str] = None,
        use_quantization: bool = False
    ):
        self.model_name = model_name
        self.device_map = device_map or "cuda:0"
        self.use_quantization = use_quantization
        self.model = None
        self.tokenizer = None
        self.model_loader = ModelLoader()
        self._loaded = False
    
    def load_model(self):
        """모델 로드"""
        if self._loaded:
            return
        
        logger.info(f"Loading LLM model: {self.model_name}")
        self.model, self.tokenizer = self.model_loader.load_llm_model(
            model_name=self.model_name,
            device_map=self.device_map,
            use_quantization=self.use_quantization
        )
        self._loaded = True
        logger.info(f"LLM model {self.model_name} loaded successfully")
    
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
        """텍스트 생성"""
        if not self._loaded:
            self.load_model()
        
        # 메시지를 프롬프트로 변환
        prompt = self._format_messages(messages)
        
        # 토크나이징
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)
        
        # 생성 파라미터
        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
        }
        
        if top_k:
            generation_config["top_k"] = top_k
        
        # stop 토큰 처리
        stop_token_ids = None
        if stop:
            stop_token_ids = []
            for stop_str in stop:
                stop_tokens = self.tokenizer.encode(stop_str, add_special_tokens=False)
                if stop_tokens:
                    stop_token_ids.extend(stop_tokens)
            if stop_token_ids:
                generation_config["eos_token_id"] = stop_token_ids
        
        # 비동기로 생성 (스레드 풀에서 실행하여 블로킹 방지)
        loop = asyncio.get_event_loop()
        
        def _generate():
            with torch.no_grad():
                return self.model.generate(
                    **inputs,
                    **generation_config
                )
        
        outputs = await loop.run_in_executor(None, _generate)
        
        # 디코딩
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # 토큰 사용량 계산
        prompt_tokens = inputs["input_ids"].shape[1]
        completion_tokens = outputs[0].shape[0] - prompt_tokens
        total_tokens = prompt_tokens + completion_tokens
        
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
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

