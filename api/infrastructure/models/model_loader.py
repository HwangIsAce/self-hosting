"""모델 로더 - Hugging Face 모델 로드 및 관리"""

from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModel
from api.config.logging_config import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """Hugging Face 모델 로더"""
    
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.loaded_tokenizers: Dict[str, Any] = {}
        self.loaded_processors: Dict[str, Any] = {}
    
    def _check_flash_attention(self) -> bool:
        """Flash Attention 2 사용 가능 여부 확인"""
        try:
            import flash_attn
            return True
        except ImportError:
            logger.warning("flash-attn not installed. Install with: pip install flash-attn")
            return False
    
    def load_llm_model(
        self,
        model_name: str,
        device_map: Optional[str] = None,
        use_quantization: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        """LLM 모델 로드"""
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded, reusing")
            return self.loaded_models[model_name], self.loaded_tokenizers[model_name]
        
        logger.info(f"Loading LLM model: {model_name}")
        
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 모델 로드 옵션
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }
        
        if device_map:
            model_kwargs["device_map"] = device_map
        else:
            model_kwargs["device_map"] = "auto"
        
        if use_quantization:
            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                model_kwargs["quantization_config"] = quantization_config
            elif load_in_8bit:
                model_kwargs["load_in_8bit"] = True
        
        # 모델 로드
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        # 평가 모드로 설정
        model.eval()
        
        self.loaded_models[model_name] = model
        self.loaded_tokenizers[model_name] = tokenizer
        
        logger.info(f"Model {model_name} loaded successfully")
        
        return model, tokenizer
    
    def load_vlm_model(
        self,
        model_name: str,
        device_map: Optional[str] = None,
        use_quantization: bool = False
    ):
        """VLM 모델 로드"""
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded, reusing")
            return self.loaded_models[model_name], self.loaded_processors[model_name]
        
        logger.info(f"Loading VLM model: {model_name}")
        
        # 프로세서 로드
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 모델 로드 옵션
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }
        
        if device_map:
            model_kwargs["device_map"] = device_map
        else:
            model_kwargs["device_map"] = "auto"
        
        # Flash Attention 2 적용
        if self._check_flash_attention():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2 for VLM model")
        
        if use_quantization:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            model_kwargs["quantization_config"] = quantization_config
        
        # 모델 로드 (VLM 모델은 Qwen2VLForConditionalGeneration 사용)
        # Qwen2-VL 모델은 반드시 Qwen2VLForConditionalGeneration 사용해야 generate 메서드가 있음
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        model.eval()
        
        self.loaded_models[model_name] = model
        self.loaded_processors[model_name] = processor
        
        logger.info(f"VLM model {model_name} loaded successfully")
        
        return model, processor
    
    def load_ocr_model(
        self,
        model_name: str,
        device_map: Optional[str] = None
    ):
        """OCR 모델 로드 (Chandra OCR)"""
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded, reusing")
            return self.loaded_models[model_name], self.loaded_processors.get(model_name)
        
        logger.info(f"Loading OCR model: {model_name}")
        
        # Chandra OCR은 Qwen3VLForConditionalGeneration 사용
        # Hugging Face 문서: https://huggingface.co/datalab-to/chandra
        # chandra.model.hf에서 Qwen3VLForConditionalGeneration 사용 확인
        try:
            from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
            
            processor = Qwen3VLProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        except ImportError:
            # Qwen3VLForConditionalGeneration이 없는 경우 AutoProcessor 사용
            try:
                processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
            except Exception as e:
                logger.warning(f"Failed to load processor: {e}")
                processor = None
        except Exception as e:
            logger.warning(f"Failed to load Qwen3VLProcessor: {e}")
            try:
                processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
            except Exception as e2:
                logger.warning(f"Failed to load AutoProcessor: {e2}")
                processor = None
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }
        
        if device_map:
            model_kwargs["device_map"] = device_map
        else:
            model_kwargs["device_map"] = "auto"
        
        # Flash Attention 2 적용
        if self._check_flash_attention():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2 for OCR model")
        
        # Chandra는 Qwen3VLForConditionalGeneration 사용 (chandra.model.hf 참조)
        try:
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                **model_kwargs
            )
        except ImportError:
            # Qwen3VLForConditionalGeneration이 없는 경우 AutoModelForCausalLM 시도
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
            except Exception:
                # 실패 시 AutoModel 사용
                model = AutoModel.from_pretrained(
                    model_name,
                    **model_kwargs
                )
        except Exception as e:
            logger.warning(f"Failed to load Qwen3VLForConditionalGeneration: {e}, trying AutoModelForCausalLM")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
            except Exception:
                model = AutoModel.from_pretrained(
                    model_name,
                    **model_kwargs
                )
        
        # processor를 model에 할당 (chandra 패키지 사용 시 필요)
        if processor:
            model.processor = processor
        
        model.eval()
        
        self.loaded_models[model_name] = model
        if processor:
            self.loaded_processors[model_name] = processor
        
        logger.info(f"OCR model {model_name} loaded successfully")
        
        return model, processor
    
    def unload_model(self, model_name: str):
        """모델 언로드 (메모리 해제)"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if model_name in self.loaded_tokenizers:
                del self.loaded_tokenizers[model_name]
            if model_name in self.loaded_processors:
                del self.loaded_processors[model_name]
            torch.cuda.empty_cache()
            logger.info(f"Model {model_name} unloaded")
    
    def get_model(self, model_name: str):
        """로드된 모델 반환"""
        return self.loaded_models.get(model_name)
    
    def get_tokenizer(self, model_name: str):
        """로드된 토크나이저 반환"""
        return self.loaded_tokenizers.get(model_name)
    
    def get_processor(self, model_name: str):
        """로드된 프로세서 반환"""
        return self.loaded_processors.get(model_name)

