"""Docling 실행 엔진 - 문서 처리"""

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
from api.config.logging_config import get_logger

logger = get_logger(__name__)


class DoclingEngine:
    """Docling 문서 처리 엔진"""
    
    def __init__(self):
        # Docling DocumentConverter 초기화
        # 옵션 설정: OCR 활성화, 표 추출 등
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True  # OCR 활성화
        pipeline_options.do_table_structure = True  # 표 구조 추출
        pipeline_options.table_structure_options.do_cell_matching = True
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        self._initialized = True
        logger.info("Docling engine initialized with OCR and table extraction enabled")
    
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
            
            # Docling으로 문서 변환 (동기 함수를 비동기로 실행)
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._convert_document,
                temp_file,
                options
            )
            
            return result
            
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
    
    async def _process_txt(self, file_data: bytes, options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """TXT 처리 (Docling이 지원하지 않으므로 직접 처리)"""
        try:
            text = file_data.decode("utf-8")
        except:
            try:
                text = file_data.decode("latin-1")
            except:
                text = file_data.decode("utf-8", errors="ignore")
        
        return {
            "text": text,
            "metadata": {
                "file_type": "txt"
            },
            "structure": {}
        }
    
    def _convert_document(
        self,
        file_path: str,
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Docling으로 문서 변환 (동기 함수)"""
        try:
            # 문서 변환
            result = self.converter.convert(file_path)
            
            # 변환 결과 추출
            doc = result.document
            
            # Markdown으로 내보내기
            markdown_text = doc.export_to_markdown()
            
            # 메타데이터 수집
            metadata = {
                "file_type": Path(file_path).suffix[1:] if Path(file_path).suffix else "unknown",
                "pages": len(doc.pages) if hasattr(doc, 'pages') and doc.pages else 1,
            }
            
            # 문서 제목이 있으면 추가
            if hasattr(doc, 'title') and doc.title:
                metadata["title"] = doc.title
            
            # 구조 정보 수집
            structure = {}
            
            # 표 정보 추출
            if hasattr(doc, 'tables') and doc.tables:
                structure["tables"] = len(doc.tables)
            
            # 이미지 정보 추출
            if hasattr(doc, 'images') and doc.images:
                structure["images"] = len(doc.images)
            
            return {
                "text": markdown_text,
                "metadata": metadata,
                "structure": structure
            }
            
        except Exception as e:
            logger.error(f"Docling conversion failed: {str(e)}")
            raise

