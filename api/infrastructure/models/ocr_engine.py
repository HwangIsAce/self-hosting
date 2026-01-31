"""OCR 실행 엔진 - Chandra OCR 모델 실행"""

from typing import Dict, Any, Optional
import torch
from PIL import Image
import base64
from io import BytesIO
import asyncio

from api.infrastructure.models.model_loader import ModelLoader
from api.config.logging_config import get_logger

logger = get_logger(__name__)


class OCREngine:
    """Chandra OCR 실행 엔진"""
    
    def __init__(
        self,
        model_name: str,
        device_map: Optional[str] = None
    ):
        self.model_name = model_name
        self.device_map = device_map or "cuda:2"
        self.model = None
        self.processor = None
        self.model_loader = ModelLoader()
        self._loaded = False
    
    def load_model(self):
        """모델 로드"""
        if self._loaded:
            return
        
        logger.info(f"Loading OCR model: {self.model_name}")
        self.model, self.processor = self.model_loader.load_ocr_model(
            model_name=self.model_name,
            device_map=self.device_map
        )
        self._loaded = True
        logger.info(f"OCR model {self.model_name} loaded successfully")
    
    async def process(
        self,
        image_base64: str,
        prompt_type: str = "ocr_layout",
        output_format: str = "markdown"
    ) -> Dict[str, Any]:
        """OCR 처리"""
        if not self._loaded:
            self.load_model()
        
        # Base64 이미지 디코딩
        image = await self._decode_base64_image(image_base64)
        if not image:
            raise ValueError("Failed to decode image")
        
        # 프롬프트 생성
        prompt = self._create_prompt(prompt_type, output_format)
        
        # 프로세서로 입력 준비
        if self.processor:
            # Qwen3VL processor는 text 파라미터가 필요할 수 있음
            try:
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt"
                )
            except TypeError:
                # text 파라미터가 없는 경우 images만 사용
                inputs = self.processor(
                    images=image,
                    return_tensors="pt"
                )
            
            # device로 이동
            if isinstance(inputs, dict):
                inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            else:
                inputs = inputs.to(self.model.device)
        else:
            # 프로세서가 없는 경우 직접 처리
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            inputs = {"pixel_values": transform(image).unsqueeze(0).to(self.model.device)}
        
        # OCR 추론
        loop = asyncio.get_event_loop()
        with torch.no_grad():
            if hasattr(self.model, "generate"):
                outputs = await loop.run_in_executor(
                    None,
                    lambda: self.model.generate(
                        **inputs,
                        max_new_tokens=512
                    )
                )
                
                # 디코딩
                if self.processor:
                    text = self.processor.decode(outputs[0], skip_special_tokens=True)
                else:
                    text = "OCR result"  # 기본값
            else:
                # 다른 모델 타입
                outputs = await loop.run_in_executor(
                    None,
                    lambda: self.model(**inputs)
                )
                text = "OCR result"
        
        # 출력 형식에 따라 포맷팅
        result = {
            "text": text,
            "metadata": {
                "prompt_type": prompt_type,
                "output_format": output_format
            }
        }
        
        if output_format == "markdown":
            result["markdown"] = text
        elif output_format == "html":
            result["html"] = f"<p>{text}</p>"
        elif output_format == "json":
            result["json"] = {"text": text}
        
        return result
    
    async def _decode_base64_image(self, base64_str: str) -> Optional[Image.Image]:
        """Base64 문자열에서 이미지 디코딩"""
        try:
            if "," in base64_str:
                base64_str = base64_str.split(",")[1]
            
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data))
            return image.convert("RGB")
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {str(e)}")
        return None
    
    def _create_prompt(self, prompt_type: str, output_format: str) -> str:
        """프롬프트 생성"""
        prompts = {
            "ocr_layout": f"Extract text from this image in {output_format} format.",
            "ocr_text": "Extract all text from this image.",
            "ocr_table": "Extract text from this table image.",
        }
        return prompts.get(prompt_type, prompts["ocr_layout"])

