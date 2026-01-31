"""VLM 실행 엔진 - Qwen VLM 모델 실행"""

from typing import List, Dict, Any, Optional
import torch
from PIL import Image
import requests
from io import BytesIO
import base64
import asyncio

from api.infrastructure.models.model_loader import ModelLoader
from api.config.logging_config import get_logger

logger = get_logger(__name__)


class VLMEngine:
    """Qwen VLM 실행 엔진"""
    
    def __init__(
        self,
        model_name: str,
        device_map: Optional[str] = None,
        use_quantization: bool = False
    ):
        self.model_name = model_name
        self.device_map = device_map or "cuda:1"
        self.use_quantization = use_quantization
        self.model = None
        self.processor = None
        self.model_loader = ModelLoader()
        self._loaded = False
    
    def load_model(self):
        """모델 로드"""
        if self._loaded:
            return
        
        logger.info(f"Loading VLM model: {self.model_name}")
        self.model, self.processor = self.model_loader.load_vlm_model(
            model_name=self.model_name,
            device_map=self.device_map,
            use_quantization=self.use_quantization
        )
        self._loaded = True
        logger.info(f"VLM model {self.model_name} loaded successfully")
    
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, Any]:
        """이미지-텍스트 멀티모달 생성"""
        if not self._loaded:
            self.load_model()
        
        # 이미지 추출 및 처리
        images = []
        text_parts = []
        
        for msg in messages:
            content = msg.get("content", "")
            image_url = msg.get("image_url")
            image_base64 = msg.get("image_base64")
            
            if image_url:
                # URL에서 이미지 다운로드
                image = await self._download_image(image_url)
                if image:
                    images.append(image)
                    text_parts.append(content)
            elif image_base64:
                # Base64에서 이미지 디코딩
                image = await self._decode_base64_image(image_base64)
                if image:
                    images.append(image)
                    text_parts.append(content)
            else:
                text_parts.append(content)
        
        # 텍스트 결합
        text = " ".join(text_parts)
        
        # 프로세서로 입력 준비
        if images:
            inputs = self.processor(
                text=text,
                images=images,
                return_tensors="pt",
                padding=True
            )
            # device로 이동
            if isinstance(inputs, dict):
                inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            else:
                inputs = inputs.to(self.model.device)
        else:
            # 이미지가 없는 경우 텍스트만
            inputs = self.processor(
                text=text,
                return_tensors="pt",
                padding=True
            )
            if isinstance(inputs, dict):
                inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            else:
                inputs = inputs.to(self.model.device)
        
        # 생성 파라미터
        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
        }
        
        # 비동기로 생성
        loop = asyncio.get_event_loop()
        with torch.no_grad():
            outputs = await loop.run_in_executor(
                None,
                lambda: self.model.generate(
                    **inputs,
                    **generation_config
                )
            )
        
        # 디코딩
        if hasattr(self.processor, "decode"):
            generated_text = self.processor.decode(
                outputs[0],
                skip_special_tokens=True
            )
        else:
            # 프로세서에 decode가 없는 경우 토크나이저 사용
            if hasattr(self.processor, "tokenizer"):
                generated_text = self.processor.tokenizer.decode(
                    outputs[0],
                    skip_special_tokens=True
                )
            else:
                # 기본 디코딩
                generated_text = str(outputs[0])
        
        # 토큰 사용량 계산 (대략적)
        prompt_tokens = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
        completion_tokens = len(outputs[0])
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
    
    async def _download_image(self, url: str) -> Optional[Image.Image]:
        """URL에서 이미지 다운로드"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(url, timeout=10)
            )
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content))
                return image.convert("RGB")
        except Exception as e:
            logger.error(f"Failed to download image from {url}: {str(e)}")
        return None
    
    async def _decode_base64_image(self, base64_str: str) -> Optional[Image.Image]:
        """Base64 문자열에서 이미지 디코딩"""
        try:
            # base64 문자열에서 데이터 부분만 추출
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]
            
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data))
            return image.convert("RGB")
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {str(e)}")
        return None

