"""OCR 실행 엔진 - Chandra OCR 모델 실행"""

from typing import Dict, Any, Optional
import torch
from PIL import Image
import base64
from io import BytesIO
import asyncio

from api.infrastructure.models.model_loader import ModelLoader
from api.config.logging_config import get_logger

# Chandra OCR 패키지 사용 시도
try:
    from transformers import AutoModel, AutoProcessor
    from chandra.model.hf import generate_hf
    from chandra.model.schema import BatchInputItem
    from chandra.output import parse_markdown
    HAS_CHANDRA_PACKAGE = True
except ImportError:
    HAS_CHANDRA_PACKAGE = False
    import warnings
    warnings.warn("chandra-ocr package not installed. Install with: pip install chandra-ocr")

logger = get_logger(__name__)


class OCREngine:
    """Chandra OCR 실행 엔진"""
    
    # 클래스 변수로 모델 관리 (싱글톤 패턴)
    _model = None
    _processor = None
    _model_loaded = False
    _parse_markdown = None
    
    def __init__(
        self,
        model_name: str,
        device_map: Optional[str] = None
    ):
        self.model_name = model_name
        self.device_map = device_map or "cuda:2"
        
        # 모델 초기화 (최초 1회만)
        if HAS_CHANDRA_PACKAGE:
            self._ensure_model_loaded()
    
    @classmethod
    def _ensure_model_loaded(cls):
        """Chandra 모델을 GPU에 로드 (싱글톤 패턴)"""
        if cls._model_loaded:
            return
        
        # GPU 체크
        if not torch.cuda.is_available():
            raise RuntimeError("Chandra parser requires GPU. CUDA is not available.")
        
        # Chandra 라이브러리 import 및 초기화
        if not HAS_CHANDRA_PACKAGE:
            raise ImportError("Chandra dependencies not installed. Install with: pip install chandra-ocr")
        
        try:
            from transformers import AutoProcessor
            from chandra.output import parse_markdown
            
            # Qwen3VLForConditionalGeneration 사용 (generate 메서드 필요)
            try:
                from transformers import Qwen3VLForConditionalGeneration
                cls._model = Qwen3VLForConditionalGeneration.from_pretrained("datalab-to/chandra").cuda()
            except ImportError:
                # Qwen3VLForConditionalGeneration이 없으면 AutoModelForCausalLM 시도
                from transformers import AutoModelForCausalLM
                cls._model = AutoModelForCausalLM.from_pretrained("datalab-to/chandra").cuda()
            
            cls._processor = AutoProcessor.from_pretrained("datalab-to/chandra")
            cls._model.processor = cls._processor
            
            # parse_markdown 함수 저장
            cls._parse_markdown = parse_markdown
            
            cls._model_loaded = True
            logger.info("Chandra model initialized successfully")
        except ImportError as e:
            raise ImportError(f"Chandra dependencies not installed: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    async def process(
        self,
        image_base64: str,
        prompt_type: str = "ocr_layout",
        output_format: str = "markdown"
    ) -> Dict[str, Any]:
        """OCR 처리 - 사용자 코드와 동일한 방식"""
        # Base64 이미지 디코딩
        image = await self._decode_base64_image(image_base64)
        if not image:
            raise ValueError("Failed to decode image")
        
        # Chandra OCR 패키지 사용
        if HAS_CHANDRA_PACKAGE and OCREngine._model and OCREngine._processor:
            try:
                # 1. BatchInputItem 생성
                batch = [BatchInputItem(image=image, prompt_type=prompt_type)]
                
                # 2. generate_hf로 생성 (핵심 호출!)
                def generate_ocr():
                    from chandra.model.hf import generate_hf
                    return generate_hf(batch, OCREngine._model)[0]
                    # generate_hf()가 GPU에서 OCR 수행
                    # result.raw에 원시 결과가 들어있음
                
                # 비동기로 실행
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, generate_ocr)
                
                # 3. Markdown 파싱
                markdown = OCREngine._parse_markdown(result.raw)
                # HTML 태그가 포함된 Markdown 형식
                
                # 출력 형식에 따라 텍스트 추출
                if output_format == "markdown":
                    text = markdown
                else:
                    # HTML 태그 제거하여 순수 텍스트 추출
                    import re
                    text = re.sub(r'<[^>]+>', '', markdown)
                
                logger.info(f"OCR completed. Text length: {len(text)}")
                
            except Exception as e:
                logger.error(f"Chandra OCR processing failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                text = await self._process_manual(image, prompt_type, output_format)
        else:
            # 수동 처리 (fallback)
            text = await self._process_manual(image, prompt_type, output_format)
        
        # 출력 형식에 따라 포맷팅
        result = {
            "text": text,
            "metadata": {
                "prompt_type": prompt_type,
                "output_format": output_format,
                "parser": "chandra" if HAS_CHANDRA_PACKAGE and OCREngine._model else "manual"
            }
        }
        
        if output_format == "markdown":
            result["markdown"] = text
        elif output_format == "html":
            result["html"] = text if "<" in text else f"<p>{text}</p>"
        elif output_format == "json":
            result["json"] = {"text": text}
        
        return result
    
    async def _process_manual(
        self,
        image: Image.Image,
        prompt_type: str,
        output_format: str
    ) -> str:
        """수동 OCR 처리 (fallback)"""
        # Chandra는 Qwen3VL 기반이므로 특별한 처리가 필요할 수 있음
        # 현재는 간단한 fallback만 제공
        logger.warning("Using manual OCR processing (chandra-ocr package recommended)")
        return "OCR processing requires chandra-ocr package. Please install with: pip install chandra-ocr"
    
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

