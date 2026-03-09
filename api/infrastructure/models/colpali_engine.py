"""ColPali 실행 엔진 - 문서 이미지 임베딩 (검색용). GPU 2 직렬화 적용."""

from typing import List, Any, Optional
import asyncio

from api.config.settings import settings
from api.config.logging_config import get_logger
from api.infrastructure.utils.gpu2_lock import (
    gpu2_lock,
    set_current_gpu2_engine,
    register_gpu2_engine,
    unload_other_gpu2_engines,
)

logger = get_logger(__name__)

# ColPali는 transformers 4.45+ 등에서 ColPaliForRetrieval, ColPaliProcessor 제공
try:
    from transformers import ColPaliForRetrieval, ColPaliProcessor
    HAS_COLPALI = True
except ImportError:
    HAS_COLPALI = False


def _embed_images_sync(images: List[Any], model: Any, processor: Any) -> Any:
    """동기 임베딩: 이미지 리스트 → ColPali 임베딩 (run_in_executor에서 호출)."""
    import torch
    inputs = processor(images=images).to(model.device)
    with torch.no_grad():
        embeddings = model(**inputs).embeddings
    return embeddings


class ColPaliEngine:
    """ColPali 문서 검색 엔진 (GPU 2, gpu2_lock 직렬화)."""

    def __init__(self):
        self.model_name = getattr(settings, "COLPALI_MODEL_NAME", "vidore/colpali-v1.3-hf")
        self.gpu_id = getattr(settings, "COLPALI_GPU_ID", 2)
        self.device_map = f"cuda:{self.gpu_id}"
        self._model = None
        self._processor = None
        self._loaded = False
        register_gpu2_engine("colpali", self._unload)
    
    def _unload(self) -> None:
        """GPU 2에서 ColPali 해제. 다른 엔진으로 전환 시 호출됨."""
        self._model = None
        self._processor = None
        self._loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        set_current_gpu2_engine(None)

    def _load_model(self) -> None:
        if self._loaded:
            return
        if not HAS_COLPALI:
            raise RuntimeError(
                "ColPali not available. Install with: pip install transformers>=4.45 (and ensure ColPaliForRetrieval is available)."
            )
        import torch
        logger.info(f"Loading ColPali model: {self.model_name} on {self.device_map}")
        self._model = ColPaliForRetrieval.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map=self.device_map,
        )
        self._processor = ColPaliProcessor.from_pretrained(self.model_name)
        self._model.eval()
        self._loaded = True
        logger.info(f"ColPali model {self.model_name} loaded successfully")

    async def embed(self, images: List[Any]) -> Any:
        """문서 페이지 이미지 리스트를 임베딩으로 변환. GPU 2 직렬화 락 사용."""
        self._load_model()
        # GPU 2 직렬화: 다른 엔진 언로드 후 ColPali만 실행
        async with gpu2_lock():
            unload_other_gpu2_engines(except_name="colpali")
            set_current_gpu2_engine("colpali")
            try:
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None,
                    _embed_images_sync,
                    images,
                    self._model,
                    self._processor,
                )
                return embeddings
            finally:
                set_current_gpu2_engine(None)
