#!/usr/bin/env python3
"""API 배치 엔드포인트 테스트"""

import asyncio
import sys
import time
import httpx
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.config.logging_config import setup_logging

setup_logging()

async def test_batch_api():
    """API 배치 엔드포인트 테스트"""
    print("="*70)
    print("API 배치 엔드포인트 테스트")
    print("="*70)
    
    base_url = "http://localhost:8000"
    
    # 배치 요청 생성
    batch_request = {
        "requests": [
            {
                "model": "qwen-llm-7b",
                "messages": [{"role": "user", "content": f"Say hello for request {i}."}],
                "max_tokens": 50
            }
            for i in range(10)
        ]
    }
    
    print(f"\n배치 요청 전송: {len(batch_request['requests'])}개 요청")
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        start = time.time()
        try:
            response = await client.post(
                f"{base_url}/v1/chat/completions/batch",
                json=batch_request
            )
            elapsed = time.time() - start
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 성공: {elapsed:.2f}초")
                print(f"✅ 총 요청: {result['total_requests']}개")
                print(f"✅ 성공: {result['successful_requests']}개")
                print(f"✅ 실패: {result['failed_requests']}개")
                
                # 첫 번째 응답 확인
                if result['responses']:
                    first_response = result['responses'][0]
                    print(f"\n첫 번째 응답 예시:")
                    print(f"  - ID: {first_response.get('id', 'N/A')}")
                    print(f"  - 모델: {first_response.get('model', 'N/A')}")
                    if first_response.get('choices'):
                        print(f"  - 내용: {first_response['choices'][0]['message']['content'][:100]}...")
                    if first_response.get('usage'):
                        print(f"  - 토큰: {first_response['usage']['total_tokens']}")
                
                return True
            else:
                print(f"❌ 실패: HTTP {response.status_code}")
                print(f"응답: {response.text}")
                return False
        except httpx.ConnectError:
            print(f"❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
            print(f"   서버 실행: uv run uvicorn api.main:app --host 0.0.0.0 --port 8000")
            return False
        except Exception as e:
            print(f"❌ 오류: {str(e)}")
            return False

async def main():
    print("API 배치 엔드포인트 테스트")
    print("="*70)
    print("주의: 서버가 실행 중이어야 합니다.")
    print("서버 실행: uv run uvicorn api.main:app --host 0.0.0.0 --port 8000")
    print("="*70)
    
    success = await test_batch_api()
    
    if success:
        print("\n" + "="*70)
        print("✅ API 배치 엔드포인트 테스트 성공!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("❌ API 배치 엔드포인트 테스트 실패")
        print("="*70)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

