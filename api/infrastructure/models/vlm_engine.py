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

    # GPU 1 동시 추론 방지용 세마포어 (한 번에 1개만 추론)
    _inference_semaphore = asyncio.Semaphore(1)
    _SEMAPHORE_TIMEOUT: float = 60.0

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

        try:
            await asyncio.wait_for(
                self._inference_semaphore.acquire(),
                timeout=self._SEMAPHORE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"VLM engine busy — another inference is in progress. "
                f"Try again after {self._SEMAPHORE_TIMEOUT}s."
            )
        
        try:
            return await self._generate_impl(messages, temperature, max_tokens, top_p, **kwargs)
        finally:
            self._inference_semaphore.release()

    async def _generate_impl(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        top_p: float,
        **kwargs
    ) -> Dict[str, Any]:
        """실제 추론 로직 (세마포어 내부에서 호출)"""
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
        
        # 메시지가 비어있는지 확인
        if not qwen_messages:
            raise ValueError("No valid messages provided. VLM models require at least one message with text or image content.")
        
        # qwen-vl-utils를 사용하여 메시지 처리 (권장)
        if HAS_QWEN_VL_UTILS:
            loop = asyncio.get_event_loop()
            processed_messages = await loop.run_in_executor(
                None,
                lambda: process_vision_info(qwen_messages)
            )
        else:
            processed_messages = qwen_messages
        
        # processed_messages가 None이거나 빈 리스트인지 확인
        if not processed_messages:
            raise ValueError("Failed to process messages. processed_messages is empty or None.")
        
        # 프로세서로 입력 준비
        # Qwen2-VL은 apply_chat_template을 사용
        # processor가 None인지 확인
        if self.processor is None:
            raise ValueError("Processor is not initialized. Model may not be loaded correctly.")
        
        # tokenizer의 special_tokens_map이 None일 수 있으므로 안전하게 처리
        # processor에 tokenizer 속성이 있고, special_tokens_map을 확인
        if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
            # special_tokens_map이 None이면 빈 dict로 설정
            if not hasattr(self.processor.tokenizer, 'special_tokens_map') or self.processor.tokenizer.special_tokens_map is None:
                self.processor.tokenizer.special_tokens_map = {}
            # special_tokens_map이 속성으로 존재하지 않으면 동적으로 추가
            elif not hasattr(self.processor.tokenizer, 'special_tokens_map'):
                setattr(self.processor.tokenizer, 'special_tokens_map', {})
        
        # apply_chat_template 시도
        text = None
        try:
            text = self.processor.apply_chat_template(
                processed_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except (TypeError, AttributeError, ValueError) as e:
            # special_tokens_map이 None인 경우 대체 처리
            logger.warning(f"apply_chat_template failed: {e}. Trying alternative approach.")
            # 원본 qwen_messages에서 텍스트 추출 (processed_messages 구조가 다를 수 있음)
            text_parts = []
            for msg in qwen_messages:
                if isinstance(msg, dict):
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                    elif isinstance(content, str):
                        text_parts.append(content)
            
            # processed_messages에서도 시도
            if not text_parts:
                for msg in processed_messages:
                    if msg is None:
                        continue
                    # msg가 list인 경우와 dict인 경우 모두 처리
                    if isinstance(msg, list):
                        for item in msg:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                    elif isinstance(msg, dict):
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    text_parts.append(item.get("text", ""))
                        elif isinstance(content, str):
                            text_parts.append(content)
            
            if text_parts:
                # 간단한 포맷팅으로 텍스트 구성
                text = "\n".join(text_parts)
                # VLM 모델을 위한 프롬프트 추가
                if text and not text.endswith("\n"):
                    text += "\n"
            else:
                raise ValueError("No text content found in messages after processing.")
        
        if not text:
            raise ValueError("Failed to generate text from messages.")
        
        # 이미지 추출
        image_inputs = []
        for msg in processed_messages:
            if msg is None:
                continue
            
            # msg가 list인 경우와 dict인 경우 모두 처리
            if isinstance(msg, list):
                # list인 경우 직접 처리
                for item in msg:
                    if isinstance(item, dict) and item.get("type") == "image":
                        image_path = item.get("image", "")
                        if image_path.startswith("data:"):
                            image = await self._decode_base64_image(image_path)
                            if image:
                                image_inputs.append(image)
                        elif image_path.startswith("http://") or image_path.startswith("https://"):
                            image = await self._download_image(image_path)
                            if image:
                                image_inputs.append(image)
            elif isinstance(msg, dict):
                # dict인 경우 기존 로직 사용
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
            "use_cache": True,  # KV Cache 활성화 (20-30% 속도 향상)
        }
        
        # pad_token_id 설정 (processor의 tokenizer에서 가져오기)
        if hasattr(self.processor, 'tokenizer') and self.processor.tokenizer is not None:
            tokenizer = self.processor.tokenizer
            if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                generation_config["pad_token_id"] = tokenizer.pad_token_id
            elif hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
                generation_config["pad_token_id"] = tokenizer.eos_token_id
        
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
        
        # 디코딩 - 입력 길이만큼 제거 (Qwen2-VL 문서 방식)
        input_ids = inputs["input_ids"]
        generated_ids_trimmed = outputs[0][input_ids.shape[1]:]
        
        # Qwen2-VL 문서에 따르면 processor.batch_decode 또는 processor.tokenizer.decode 사용
        # 단일 입력이므로 batch_decode를 리스트로 감싸서 사용
        try:
            # 먼저 batch_decode 시도 (Qwen2-VL 권장 방식)
            if hasattr(self.processor, 'batch_decode'):
                decoded_texts = self.processor.batch_decode(
                    [generated_ids_trimmed],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                generated_text = decoded_texts[0] if decoded_texts else ""
            # tokenizer.decode 사용 (fallback)
            elif hasattr(self.processor, 'tokenizer') and hasattr(self.processor.tokenizer, 'decode'):
                generated_text = self.processor.tokenizer.decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
            else:
                raise AttributeError("Processor does not support batch_decode or tokenizer.decode")
        except Exception as e:
            logger.error(f"Failed to decode with processor: {e}")
            # 최후의 수단: tokenizer 직접 접근
            if hasattr(self.processor, 'tokenizer'):
                generated_text = self.processor.tokenizer.decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True
                )
            else:
                raise
        
        # 토큰 사용량 계산
        prompt_tokens = input_ids.shape[1]
        completion_tokens = len(generated_ids_trimmed)
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

