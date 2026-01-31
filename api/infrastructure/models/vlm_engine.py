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

# qwen-vl-utils 사용 (선택사항이지만 권장)
try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    # logger는 아직 초기화되지 않았으므로 print 사용
    import warnings
    warnings.warn("qwen-vl-utils not installed. Install with: pip install qwen-vl-utils")

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
        
        # Qwen2-VL 형식의 메시지로 변환
        # OpenAI 형식에서 Qwen2-VL 형식으로 변환
        qwen_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            image_url = msg.get("image_url")
            image_base64 = msg.get("image_base64")
            
            qwen_content = []
            
            # 이미지 처리
            if image_url:
                # URL 형식: http:// 또는 https://
                if HAS_QWEN_VL_UTILS:
                    qwen_content.append({"type": "image", "image": image_url})
                else:
                    # qwen-vl-utils가 없으면 다운로드 후 base64로 변환
                    image = await self._download_image(image_url)
                    if image:
                        buffer = BytesIO()
                        image.save(buffer, format='PNG')
                        image_base64 = base64.b64encode(buffer.getvalue()).decode()
                        qwen_content.append({"type": "image", "image": f"data:image/png;base64,{image_base64}"})
            elif image_base64:
                # Base64 형식
                if not image_base64.startswith("data:"):
                    image_base64 = f"data:image/png;base64,{image_base64}"
                qwen_content.append({"type": "image", "image": image_base64})
            
            # 텍스트 추가
            if content:
                qwen_content.append({"type": "text", "text": content})
            
            if qwen_content:
                qwen_messages.append({
                    "role": role,
                    "content": qwen_content
                })
        
        # qwen-vl-utils를 사용하여 메시지 처리 (권장)
        if HAS_QWEN_VL_UTILS:
            loop = asyncio.get_event_loop()
            processed_messages = await loop.run_in_executor(
                None,
                lambda: process_vision_info(qwen_messages)
            )
        else:
            processed_messages = qwen_messages
        
        # 프로세서로 입력 준비
        # Qwen2-VL은 apply_chat_template을 사용
        text = self.processor.apply_chat_template(
            processed_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # 이미지 추출
        image_inputs = []
        for msg in processed_messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        image_path = item.get("image", "")
                        if image_path.startswith("data:"):
                            # Base64 이미지 디코딩
                            image = await self._decode_base64_image(image_path)
                            if image:
                                image_inputs.append(image)
                        elif image_path.startswith("http://") or image_path.startswith("https://"):
                            # URL 이미지 다운로드
                            image = await self._download_image(image_path)
                            if image:
                                image_inputs.append(image)
                        elif image_path.startswith("file://"):
                            # 로컬 파일 (현재는 지원하지 않음)
                            logger.warning(f"Local file path not supported: {image_path}")
        
        # 프로세서로 입력 준비
        if image_inputs:
            inputs = self.processor(
                text=text,
                images=image_inputs,
                padding=True,
                return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=text,
                padding=True,
                return_tensors="pt"
            )
        
        # device로 이동
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
        
        # 디코딩 - 입력 길이만큼 제거
        input_ids = inputs["input_ids"]
        generated_ids = outputs[0][input_ids.shape[1]:]
        
        # 프로세서로 디코딩
        generated_text = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        # 토큰 사용량 계산
        prompt_tokens = input_ids.shape[1]
        completion_tokens = len(generated_ids)
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

