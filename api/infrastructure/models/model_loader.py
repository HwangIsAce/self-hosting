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
        
        if use_quantization:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16
            )
            model_kwargs["quantization_config"] = quantization_config
        
        # 모델 로드 (VLM 모델은 AutoModel로 자동 감지)
        try:
            # 먼저 AutoModelForCausalLM 시도 (일부 VLM 모델)
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
        """OCR 모델 로드"""
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded, reusing")
            return self.loaded_models[model_name], self.loaded_processors.get(model_name)
        
        logger.info(f"Loading OCR model: {model_name}")
        
        # OCR 모델은 일반적으로 Vision 모델이거나 특수 모델
        # Chandra의 경우 AutoModelForVision2Seq 또는 AutoModelForImageClassification 사용 가능
        try:
            processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        except:
            processor = None
        
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16,
        }
        
        if device_map:
            model_kwargs["device_map"] = device_map
        else:
            model_kwargs["device_map"] = "auto"
        
        # OCR 모델 로드 (모델 타입에 따라 다를 수 있음)
        try:
            # 먼저 AutoModelForCausalLM 시도
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

