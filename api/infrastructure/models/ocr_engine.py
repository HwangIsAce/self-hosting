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
    
    @classmethod
    def _reset_model(cls):
        """모델 리셋 (재로드용)"""
        cls._model = None
        cls._processor = None
        cls._model_loaded = False
        cls._parse_markdown = None
    
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
        """Chandra 모델을 GPU에 로드 (싱글톤 패턴) - Hugging Face 공식 문서 방식"""
        if cls._model_loaded:
            return
        
        # GPU 체크
        if not torch.cuda.is_available():
            raise RuntimeError("Chandra parser requires GPU. CUDA is not available.")
        
        # Chandra 라이브러리 import 및 초기화
        if not HAS_CHANDRA_PACKAGE:
            raise ImportError("Chandra dependencies not installed. Install with: pip install chandra-ocr")
        
        try:
            from transformers import AutoModel, AutoProcessor
            from chandra.output import parse_markdown
            
            # device_map을 사용하여 특정 GPU에 로드
            # 클래스 변수로 device_map 저장 (나중에 사용)
            if not hasattr(cls, '_device_map'):
                # 기본값은 cuda:2 (OCR_GPU_ID)
                from api.config.settings import settings
                cls._device_map = f"cuda:{settings.OCR_GPU_ID}"
            
            # Hugging Face 공식 문서 방식 그대로 사용
            # https://huggingface.co/datalab-to/chandra
            # model = AutoModel.from_pretrained("datalab-to/chandra").cuda()
            # 모델을 설정된 GPU에 로드
            # device_map을 사용하거나 .cuda() 메서드 사용
            device = torch.device(cls._device_map)
            cls._model = AutoModel.from_pretrained(
                "datalab-to/chandra",
                torch_dtype=torch.float16
            ).to(device)
            
            # AutoModel이 generate 메서드를 가지는지 확인
            if not hasattr(cls._model, 'generate'):
                logger.warning("AutoModel does not have generate, using Qwen3VLForConditionalGeneration")
                from transformers import Qwen3VLForConditionalGeneration
                cls._model = Qwen3VLForConditionalGeneration.from_pretrained(
                    "datalab-to/chandra",
                    torch_dtype=torch.float16
                ).to(device)
            
            # model.processor = AutoProcessor.from_pretrained("datalab-to/chandra")
            # 공식 문서에 따라 processor를 모델에 할당
            cls._processor = AutoProcessor.from_pretrained("datalab-to/chandra")
            cls._model.processor = cls._processor
            
            # processor의 디바이스를 모델과 같은 디바이스로 설정
            # generate_hf 함수가 processor를 사용할 때 올바른 디바이스를 사용하도록
            if hasattr(cls._processor, 'device'):
                cls._processor.device = device
            # processor 내부의 tokenizer나 다른 컴포넌트도 디바이스 설정
            if hasattr(cls._processor, 'tokenizer') and hasattr(cls._processor.tokenizer, 'model'):
                try:
                    cls._processor.tokenizer.model = None  # tokenizer는 일반적으로 CPU에서 실행
                except:
                    pass
            
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
        output_format: str = "markdown",
        max_tokens: int = 1024  # OCR에 적합한 기본값
    ) -> Dict[str, Any]:
        """OCR 처리 - 사용자 코드와 동일한 방식"""
        # Base64 이미지 디코딩
        image = await self._decode_base64_image(image_base64)
        if not image:
            raise ValueError("Failed to decode image")
        
        # Chandra OCR 패키지 사용
        if HAS_CHANDRA_PACKAGE and OCREngine._model and OCREngine._processor:
            try:
                # 1. BatchInputItem 생성 (Hugging Face 공식 문서 방식)
                batch = [
                    BatchInputItem(
                        image=image,
                        prompt_type=prompt_type
                    )
                ]
                
                # 2. Hugging Face 공식 문서 방식: generate_hf 직접 사용
                # https://huggingface.co/datalab-to/chandra
                # result = generate_hf(batch, model)[0]
                def generate_ocr():
                    from chandra.model.hf import generate_hf
                    import chandra.model.hf as chandra_hf_module
                    
                    model = OCREngine._model
                    processor = OCREngine._processor
                    model_device = next(model.parameters()).device
                    logger.info(f"Using generate_hf with model type: {type(model).__name__}, device: {model_device}")
                    
                    # generate_hf 함수가 내부적으로 inputs.to("cuda")를 사용하므로,
                    # 모델의 실제 디바이스를 사용하도록 패치
                    original_generate_hf = generate_hf
                    
                    def patched_generate_hf(batch, model, max_output_tokens=None, **kwargs):
                        """generate_hf를 패치해서 모델의 실제 디바이스를 사용하도록 수정"""
                        from chandra.model.schema import GenerationResult
                        from chandra.model.hf import process_batch_element, process_vision_info, settings as chandra_settings
                        
                        if max_output_tokens is None:
                            max_output_tokens = getattr(chandra_settings, 'MAX_OUTPUT_TOKENS', 1024)
                        
                        # 모델의 실제 디바이스 사용
                        device = next(model.parameters()).device
                        logger.info(f"Using device {device} for generate_hf")
                        
                        messages = [process_batch_element(item, model.processor) for item in batch]
                        text = model.processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        
                        image_inputs, _ = process_vision_info(messages)
                        inputs = model.processor(
                            text=text,
                            images=image_inputs,
                            padding=True,
                            return_tensors="pt",
                            padding_side="left",
                        )
                        # 모델의 실제 디바이스로 이동 (원래는 .to("cuda")로 하드코딩됨)
                        # inputs 딕셔너리의 모든 텐서를 명시적으로 디바이스로 이동
                        if isinstance(inputs, dict):
                            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                     for k, v in inputs.items()}
                        else:
                            inputs = inputs.to(device)
                        
                        logger.info(f"Inputs moved to device {device}, input_ids device: {inputs['input_ids'].device if 'input_ids' in inputs else 'N/A'}")
                        
                        # Inference: Generation of the output
                        # do_sample=False로 설정하여 greedy decoding 사용 (확률 분포 문제 방지)
                        generation_kwargs = {
                            "max_new_tokens": max_output_tokens,
                            "do_sample": False,  # greedy decoding 사용
                        }
                        # generation_config가 있으면 추가 설정 사용
                        if hasattr(model, 'generation_config') and model.generation_config is not None:
                            if hasattr(model.generation_config, 'pad_token_id'):
                                generation_kwargs['pad_token_id'] = model.generation_config.pad_token_id
                            if hasattr(model.generation_config, 'eos_token_id'):
                                generation_kwargs['eos_token_id'] = model.generation_config.eos_token_id
                        
                        generated_ids = model.generate(**inputs, **generation_kwargs)
                        generated_ids_trimmed = [
                            out_ids[len(in_ids) :]
                            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                        ]
                        output_text = model.processor.batch_decode(
                            generated_ids_trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        results = [
                            GenerationResult(raw=out, token_count=len(ids), error=False)
                            for out, ids in zip(output_text, generated_ids_trimmed)
                        ]
                        return results
                    
                    # max_tokens를 모델의 generation config에 설정 (있는 경우)
                    original_max_new_tokens = None
                    if hasattr(model, 'generation_config') and model.generation_config is not None:
                        original_max_new_tokens = getattr(model.generation_config, 'max_new_tokens', None)
                        model.generation_config.max_new_tokens = max_tokens
                        logger.info(f"Set max_new_tokens to {max_tokens} in generation_config")
                    
                    # 패치된 generate_hf 사용
                    try:
                        results = patched_generate_hf(batch, model, max_output_tokens=max_tokens)
                    except Exception as e:
                        logger.error(f"Patched generate_hf failed: {e}, trying original")
                        # 패치 실패 시 원본 함수 사용 (max_output_tokens 파라미터 지원 여부 확인)
                        try:
                            results = generate_hf(batch, model, max_output_tokens=max_tokens)
                        except TypeError:
                            results = generate_hf(batch, model)
                    
                    # generation config 복원 (있는 경우)
                    if hasattr(model, 'generation_config') and model.generation_config is not None and original_max_new_tokens is not None:
                        model.generation_config.max_new_tokens = original_max_new_tokens
                    
                    if not results or len(results) == 0:
                        logger.error("Empty results from generate_hf")
                        raise ValueError("Empty results from generate_hf")
                    
                    result = results[0]  # 첫 번째 결과 사용
                    
                    logger.info(f"Generated result.raw length: {len(result.raw) if result.raw else 0}")
                    if result.raw:
                        logger.info(f"Generated result.raw (first 200 chars): {result.raw[:200]}")
                    
                    return result
                
                # 비동기로 실행
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, generate_ocr)
                
                # 디버깅: raw 출력 확인
                logger.info(f"OCR raw output length: {len(result.raw) if result.raw else 0}")
                logger.info(f"OCR raw output (first 500 chars): {result.raw[:500] if result.raw else 'None'}")
                
                # 3. Markdown 파싱 (공식 문서 방식)
                # markdown = parse_markdown(result.raw)
                if not result.raw or not result.raw.strip():
                    logger.warning("Empty raw output from model generation")
                    markdown = ""
                else:
                    try:
                        if OCREngine._parse_markdown:
                            markdown = OCREngine._parse_markdown(result.raw)
                            logger.info(f"Parsed markdown length: {len(markdown) if markdown else 0}")
                        else:
                            logger.warning("_parse_markdown function not available, using raw output")
                            markdown = result.raw
                    except Exception as e:
                        logger.error(f"Error parsing markdown: {e}")
                        # 파싱 실패 시 raw 출력을 그대로 사용
                        markdown = result.raw
                
                # HTML 태그가 포함된 Markdown 형식
                
                # 출력 형식에 따라 텍스트 추출
                if output_format == "markdown":
                    text = markdown
                else:
                    # HTML 태그 제거하여 순수 텍스트 추출
                    import re
                    text = re.sub(r'<[^>]+>', '', markdown) if markdown else ""
                
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

