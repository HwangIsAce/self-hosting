"""Chandra OCR 엔진 테스트 스크립트"""

import asyncio
import sys
import os
from pathlib import Path
import base64

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.infrastructure.models.ocr_engine import OCREngine
from api.config.settings import settings
from api.config.logging_config import setup_logging, get_logger

# 로깅 설정
setup_logging()
logger = get_logger(__name__)


async def test_chandra_ocr():
    """Chandra OCR 엔진 테스트"""
    print("=" * 60)
    print("Chandra OCR 엔진 테스트 시작")
    print("=" * 60)
    
    # 1. GPU 확인
    try:
        import torch
        if not torch.cuda.is_available():
            print("❌ CUDA가 사용 불가능합니다. GPU가 필요합니다.")
            return False
        print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch가 설치되지 않았습니다.")
        return False
    
    # 2. Chandra 패키지 확인
    try:
        from chandra.model import InferenceManager
        print("✅ chandra-ocr 패키지 설치됨")
    except ImportError:
        print("❌ chandra-ocr 패키지가 설치되지 않았습니다.")
        print("   설치: pip install chandra-ocr")
        return False
    
    # 3. OCREngine 초기화
    print("\n[1/4] OCREngine 초기화 중...")
    try:
        engine = OCREngine(
            model_name=settings.OCR_MODEL_NAME,
            device_map=f"cuda:{settings.OCR_GPU_ID}"
        )
        print(f"✅ OCREngine 초기화 완료 (모델: {settings.OCR_MODEL_NAME}, GPU: {settings.OCR_GPU_ID})")
    except Exception as e:
        print(f"❌ OCREngine 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 테스트 이미지 준비
    print("\n[2/4] 테스트 이미지 준비 중...")
    # 한글 파일명 처리
    import os
    test_image_dir = project_root / "docs" / "input"
    test_image_path = None
    
    # 디렉토리에서 파일 찾기 (peterparser.png 우선)
    if test_image_dir.exists():
        # 먼저 peterparser.png 찾기
        peterparser_path = test_image_dir / "peterparser.png"
        if peterparser_path.exists():
            test_image_path = peterparser_path
        else:
            # 없으면 첫 번째 .png 파일 사용
            for file in os.listdir(str(test_image_dir)):
                if file.endswith('.png'):
                    test_image_path = test_image_dir / file
                    break
    
    if not test_image_path or not test_image_path.exists():
        print(f"❌ 테스트 이미지가 없습니다: {test_image_dir}")
        return False
    
    print(f"✅ 테스트 이미지 찾음: {test_image_path.name}")
    
    # 이미지를 base64로 인코딩
    try:
        with open(test_image_path, 'rb') as f:
            image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        print(f"✅ 이미지 로드 완료: {test_image_path} ({len(image_data)} bytes)")
    except Exception as e:
        print(f"❌ 이미지 로드 실패: {e}")
        return False
    
    # 5. OCR 처리
    print("\n[3/4] OCR 처리 중... (시간이 걸릴 수 있습니다)")
    try:
        result = await engine.process(
            image_base64=image_base64,
            prompt_type="ocr_layout",
            output_format="markdown",
            max_tokens=256  # 테스트 속도 향상을 위해 토큰 수 감소
        )
        
        print("✅ OCR 처리 완료")
        print(f"\n[4/4] 결과:")
        print("-" * 60)
        print(f"Parser: {result['metadata'].get('parser', 'unknown')}")
        print(f"Text length: {len(result.get('text', ''))}")
        print(f"\n추출된 텍스트 (처음 500자):")
        print(result.get('text', '')[:500])
        if len(result.get('text', '')) > 500:
            print("...")
        print("-" * 60)
        
        # Markdown 결과도 확인
        if 'markdown' in result:
            print(f"\nMarkdown 결과 (처음 500자):")
            print(result['markdown'][:500])
            if len(result['markdown']) > 500:
                print("...")
        
        return True
        
    except Exception as e:
        print(f"❌ OCR 처리 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """메인 함수"""
    success = await test_chandra_ocr()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 테스트 완료: Chandra OCR가 정상적으로 동작합니다!")
    else:
        print("❌ 테스트 실패: 문제를 확인해주세요.")
    print("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

