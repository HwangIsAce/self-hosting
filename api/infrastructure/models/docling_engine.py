"""Docling 실행 엔진 - 문서 처리 (GPU 가속 지원)"""

from typing import Dict, Any, Optional
import base64
from io import BytesIO
import asyncio
import tempfile
import os
from pathlib import Path

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from api.config.settings import settings
from api.config.logging_config import get_logger
from api.infrastructure.utils.gpu2_lock import (
    gpu2_lock,
    set_current_gpu2_engine,
    register_gpu2_engine,
    unload_other_gpu2_engines,
)

logger = get_logger(__name__)


def _json_safe_string(s: str) -> str:
    """JSON 직렬화 시 'unterminated string literal' 등을 방지하기 위해 문자열 보정.
    널 바이트·제어 문자 제거, UTF-8 정규화."""
    if not isinstance(s, str):
        return str(s)
    s = s.replace("\x00", "")
    return s.encode("utf-8", errors="replace").decode("utf-8")


def _json_safe_value(v: Any) -> Any:
    """dict/list 내부의 모든 문자열을 JSON-safe하게 보정."""
    if isinstance(v, str):
        return _json_safe_string(v)
    if isinstance(v, dict):
        return {k: _json_safe_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_json_safe_value(x) for x in v]
    return v


def _build_docling_pipeline_options():
    """Docling 파이프라인 옵션 생성 (GPU 사용 시 AcceleratorOptions 및 배치 크기 적용)"""
    use_gpu = getattr(settings, "DOCLING_USE_GPU", False)
    gpu_id = getattr(settings, "DOCLING_GPU_ID", 2)
    ocr_batch = getattr(settings, "DOCLING_OCR_BATCH_SIZE", 16)
    layout_batch = getattr(settings, "DOCLING_LAYOUT_BATCH_SIZE", 16)

    # GPU 사용 시 ThreadedPdfPipelineOptions로 배치 크기 설정 (선택)
    try:
        from docling.datamodel.accelerator_options import AcceleratorOptions
        import torch

        if use_gpu and torch.cuda.is_available():
            accelerator_options = AcceleratorOptions(
                device=f"cuda:{gpu_id}",
                num_threads=4,
            )
            try:
                from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
                pipeline_options = ThreadedPdfPipelineOptions(
                    ocr_batch_size=ocr_batch,
                    layout_batch_size=layout_batch,
                )
                logger.info(
                    f"Docling GPU enabled: cuda:{gpu_id}, ocr_batch_size={ocr_batch}, layout_batch_size={layout_batch}"
                )
            except ImportError:
                pipeline_options = PdfPipelineOptions()
                logger.info(f"Docling GPU enabled: cuda:{gpu_id} (ThreadedPdfPipelineOptions unavailable)")
            pipeline_options.accelerator_options = accelerator_options
        else:
            pipeline_options = PdfPipelineOptions()
            if use_gpu and not torch.cuda.is_available():
                logger.warning("Docling USE_GPU=True but CUDA not available, using CPU")
    except ImportError as e:
        logger.warning(f"Docling accelerator options unavailable: {e}, using CPU")
        pipeline_options = PdfPipelineOptions()

    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    return pipeline_options


def _build_docling_pipeline_options_cpu():
    """GPU 없이 CPU 전용 파이프라인 (meta tensor 에러 시 폴백용)"""
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    return pipeline_options


class DoclingEngine:
    """Docling 문서 처리 엔진 (GPU 가속 지원). 한 시점에 한 모델만 GPU 2에 로드."""
    
    def __init__(self):
        self._pipeline_options = _build_docling_pipeline_options()
        self._pipeline_options_cpu = _build_docling_pipeline_options_cpu()
        self.converter = None  # lazy init inside lock
        self._use_cpu_fallback = False  # meta tensor 발생 시 True로 전환
        self._initialized = True
        register_gpu2_engine("docling", self._unload)
    
    def _unload(self) -> None:
        """GPU 2에서 Docling 해제. 다른 엔진으로 전환 시 호출됨."""
        self.converter = None
        try:
            import torch
            gpu_id = getattr(settings, "DOCLING_GPU_ID", 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        set_current_gpu2_engine(None)
    
    async def initialize(self):
        """Docling 초기화 (이미 초기화됨)"""
        pass
    
    async def process_document(
        self,
        file_base64: str,
        file_type: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """문서 처리 - Docling을 사용하여 실제 문서 변환"""
        options = options or {}
        
        # Base64 디코딩
        try:
            file_data = base64.b64decode(file_base64)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 file: {str(e)}")
        
        # 파일 확장자 결정
        if file_type:
            ext = file_type.lower()
        else:
            # 기본값: PDF로 가정
            ext = "pdf"
        
        # TXT 파일은 Docling이 지원하지 않으므로 별도 처리
        if ext in ["txt", "text"]:
            return await self._process_txt(file_data, options)
        
        # Docling이 지원하는 형식인지 확인
        docling_supported = {
            "pdf", "docx", "pptx", "html", "htm", 
            "xlsx", "csv", "md", "asciidoc", "image"
        }
        
        if ext not in docling_supported:
            logger.warning(f"File type {ext} not directly supported by Docling, trying as generic")
        
        # 임시 파일 생성
        temp_file = None
        try:
            # 적절한 확장자로 임시 파일 생성
            suffix_map = {
                "pdf": ".pdf",
                "docx": ".docx",
                "pptx": ".pptx",
                "html": ".html",
                "htm": ".html",
                "xlsx": ".xlsx",
                "csv": ".csv",
                "md": ".md",
                "asciidoc": ".adoc",
                "image": ".png",  # 이미지는 PNG로 가정
            }
            suffix = suffix_map.get(ext, ".pdf")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_data)
                temp_file = tmp.name
            
            logger.info(f"Processing document with Docling: type={ext}, size={len(file_data)} bytes")
            
            # GPU 2 직렬화: 락 획득 후 다른 엔진 언로드, Docling만 실행
            async with gpu2_lock():
                unload_other_gpu2_engines(except_name="docling")
                set_current_gpu2_engine("docling")
                try:
                    result = await self._run_conversion(temp_file, options)
                    return result
                except NotImplementedError as e:
                    if "meta tensor" not in str(e).lower() and "to_empty" not in str(e).lower():
                        raise
                    # CPU 폴백 1회 시도 (재귀 없음)
                    logger.warning(
                        "Docling GPU pipeline failed (meta tensor), retrying with CPU: %s",
                        str(e)[:200],
                    )
                    self.converter = None
                    self._use_cpu_fallback = True
                    result = await self._run_conversion(temp_file, options)
                    return result
                # 엔진 식별자를 "docling"으로 유지 (연속 문서 요청 시 재로드 방지)
            
        except Exception as e:
            logger.exception(f"Error processing document: {str(e)}")
            raise ValueError(f"Failed to process document: {str(e)}")
        finally:
            # 임시 파일 삭제
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to delete temp file: {e}")
    
    async def _run_conversion(self, temp_file: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """락 내부에서 converter 초기화 + 변환 실행. 매 호출마다 새 converter 생성."""
        pipeline_opts = self._pipeline_options_cpu if self._use_cpu_fallback else self._pipeline_options
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
            }
        )
        logger.info(
            "Docling converter created (%s)",
            "CPU fallback" if self._use_cpu_fallback else "GPU",
        )
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._convert_document,
            temp_file,
            options,
        )

    async def _process_txt(self, file_data: bytes, options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """TXT 처리 (Docling이 지원하지 않으므로 직접 처리) - markdown, html, json 모두 반환"""
        try:
            text = file_data.decode("utf-8")
        except Exception:
            try:
                text = file_data.decode("latin-1")
            except Exception:
                text = file_data.decode("utf-8", errors="ignore")
        import html as html_module
        escaped = html_module.escape(text)
        return {
            "text": text,
            "markdown": text,
            "html": f"<pre>{escaped}</pre>",
            "json": {"text": text, "metadata": {"file_type": "txt"}},
            "metadata": {"file_type": "txt"},
            "structure": {},
        }
    
    def _convert_document(
        self,
        file_path: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Docling으로 문서 변환 (동기 함수) - 항상 markdown, html, json 반환"""
        try:
            result = self.converter.convert(file_path)
            doc = result.document

            markdown_text = doc.export_to_markdown()
            markdown_text = _json_safe_string(markdown_text or "")

            # HTML 내보내기 (DoclingDocument.export_to_html)
            html_text = ""
            if hasattr(doc, "export_to_html") and callable(doc.export_to_html):
                try:
                    html_text = doc.export_to_html()
                except Exception as e:
                    logger.warning(f"Docling export_to_html failed: {e}, using fallback")
                    html_text = f"<pre>{markdown_text}</pre>" if markdown_text else ""
            else:
                html_text = f"<pre>{markdown_text}</pre>" if markdown_text else ""
            html_text = _json_safe_string(html_text)

            # JSON 내보내기 (DoclingDocument.export_to_dict)
            json_data = {}
            if hasattr(doc, "export_to_dict") and callable(doc.export_to_dict):
                try:
                    json_data = doc.export_to_dict()
                except Exception as e:
                    logger.warning(f"Docling export_to_dict failed: {e}, using fallback")
                    json_data = {"text": markdown_text, "markdown": markdown_text}
            else:
                json_data = {"text": markdown_text, "markdown": markdown_text}
            json_data = _json_safe_value(json_data)

            metadata = {
                "file_type": Path(file_path).suffix[1:] if Path(file_path).suffix else "unknown",
                "pages": len(doc.pages) if hasattr(doc, "pages") and doc.pages else 1,
            }
            if hasattr(doc, "title") and doc.title:
                metadata["title"] = doc.title

            structure = {}
            if hasattr(doc, "tables") and doc.tables:
                structure["tables"] = len(doc.tables)
            if hasattr(doc, "pictures") and doc.pictures:
                structure["pictures"] = len(doc.pictures)
            elif hasattr(doc, "images") and doc.images:
                structure["images"] = len(doc.images)

            return {
                "text": markdown_text,
                "markdown": markdown_text,
                "html": html_text,
                "json": json_data,
                "metadata": metadata,
                "structure": structure,
            }
        except Exception as e:
            logger.error(f"Docling conversion failed: {str(e)}")
            raise

