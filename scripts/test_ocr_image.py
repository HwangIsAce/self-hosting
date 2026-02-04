#!/usr/bin/env python3
"""OCR 이미지 테스트 스크립트"""

import os
import sys
import requests
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_ocr_image(image_filename: str):
    """OCR 이미지 테스트"""
    # 디렉토리에서 파일 찾기
    input_dir = project_root / "docs" / "input"
    
    # 모든 파일 확인
    files = list(input_dir.glob("*.png"))
    print(f"찾은 PNG 파일: {[f.name for f in files]}")
    
    # 대상 파일 찾기
    target_file = None
    for f in files:
        if image_filename in f.name or f.name == image_filename:
            target_file = f
            break
    
    if not target_file:
        print(f"파일을 찾을 수 없습니다: {image_filename}")
        return False
    
    print(f"\n사용할 파일: {target_file}")
    print(f"파일 크기: {target_file.stat().st_size / 1024:.2f} KB")
    
    # 파일 읽기
    with open(target_file, 'rb') as f:
        image_content = f.read()
    
    # API 호출
    url = 'http://localhost:8000/v1/ocr'
    
    files_data = {
        'image': (target_file.name, image_content, 'image/png')
    }
    form_data = {
        'prompt_type': 'ocr_layout',
        'output_format': 'markdown',
        'max_tokens': 2048
    }
    
    print('\nOCR API 호출 중... (시간이 걸릴 수 있습니다)')
    try:
        response = requests.post(url, files=files_data, data=form_data, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            print('\n=== OCR 결과 ===')
            print(f'ID: {result.get("id")}')
            print(f'텍스트 길이: {len(result.get("text", ""))}')
            print(f'\n추출된 텍스트:\n')
            text = result.get('text', '')
            print(text)
            if len(text) > 3000:
                print(f'\n... (총 {len(text)}자)')
            return True
        else:
            print(f'에러: {response.status_code}')
            print(response.text)
            return False
    except Exception as e:
        print(f'에러 발생: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    image_filename = sys.argv[1] if len(sys.argv) > 1 else "진료영수증.png"
    success = test_ocr_image(image_filename)
    sys.exit(0 if success else 1)

