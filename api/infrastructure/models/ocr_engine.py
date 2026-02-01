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
        # GPU 2 사용 (설정에 따라 OCR_GPU_ID=2)
        # settings에서 OCR_GPU_ID를 읽어와서 사용
        from api.config.settings import settings
        self.device_map = device_map or f"cuda:{settings.OCR_GPU_ID}"
        
        # device_map을 클래스 변수에 저장 (모델 로드 시 사용)
        OCREngine._device_map = self.device_map
        
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
            
            # device_map을 사용하여 특정 GPU에 로드
            # 클래스 변수로 device_map 저장 (나중에 사용)
            if not hasattr(cls, '_device_map'):
                # 기본값은 cuda:2 (OCR_GPU_ID)
                cls._device_map = "cuda:2"
            
            # Qwen3VLForConditionalGeneration 사용 (generate 메서드 필요)
            try:
                from transformers import Qwen3VLForConditionalGeneration
                # device_map을 사용하여 특정 GPU에 로드
                cls._model = Qwen3VLForConditionalGeneration.from_pretrained(
                    "datalab-to/chandra",
                    device_map=cls._device_map,
                    torch_dtype=torch.float16  # 메모리 절약을 위해 float16 사용
                )
            except ImportError:
                # Qwen3VLForConditionalGeneration이 없으면 AutoModelForCausalLM 시도
                from transformers import AutoModelForCausalLM
                cls._model = AutoModelForCausalLM.from_pretrained(
                    "datalab-to/chandra",
                    device_map=cls._device_map,
                    torch_dtype=torch.float16
                )
            
            cls._processor = AutoProcessor.from_pretrained("datalab-to/chandra")
            cls._model.processor = cls._processor
            
            # processor의 device 설정 (모델과 같은 device)
            if hasattr(cls._processor, 'device'):
                cls._processor.device = torch.device(cls._device_map)
            
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
                
                # 2. generate_hf 내부 로직을 직접 구현하여 device를 올바르게 설정
                def generate_ocr():
                    from chandra.model.hf import process_batch_element
                    from chandra.model.schema import GenerationResult
                    from qwen_vl_utils import process_vision_info
                    import torch
                    
                    # 모델과 processor 가져오기
                    model = OCREngine._model
                    processor = OCREngine._processor
                    model_device = next(model.parameters()).device
                    
                    # process_batch_element로 메시지 생성
                    messages = [process_batch_element(item, processor) for item in batch]
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    
                    # vision 정보 처리 (chandra의 generate_hf와 동일하게 2개만 받음)
                    image_inputs, _ = process_vision_info(messages)
                    
                    # processor로 입력 준비
                    inputs = processor(
                        text=text,
                        images=image_inputs,
                        padding=True,
                        return_tensors="pt",
                        padding_side="left",
                    )
                    
                    # 모든 텐서를 올바른 device로 이동
                    inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v 
                             for k, v in inputs.items()}
                    
                    # 생성 (기본 max_output_tokens는 2048로 설정)
                    max_output_tokens = 2048
                    # generation config 설정 (Qwen3VL에 맞게)
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_output_tokens,
                        do_sample=False,  # greedy decoding
                    )
                    
                    # 출력 처리
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):]
                        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                    
                    # GenerationResult 생성
                    if not output_text or not output_text[0]:
                        raise ValueError("Empty output from model generation")
                    
                    result = GenerationResult(
                        raw=output_text[0], 
                        token_count=len(generated_ids_trimmed[0]), 
                        error=False
                    )
                    
                    return result
                
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
                error_trace = traceback.format_exc()
                logger.error(error_trace)
                # CUDA 에러인 경우 더 명확한 메시지
                if "CUDA" in str(e) or "cuda" in str(e).lower():
                    raise RuntimeError(
                        f"CUDA error during OCR processing: {str(e)}. "
                        f"Model device: {next(OCREngine._model.parameters()).device if OCREngine._model else 'unknown'}. "
                        f"Please check GPU memory and device allocation."
                    )
                # 다른 에러는 그대로 전파
                raise
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

