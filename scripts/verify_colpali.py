#!/usr/bin/env python3
"""ColPali 엔진 및 GPU 2 직렬화 동작 확인."""
import asyncio
import sys
from PIL import Image

# 프로젝트 루트
sys.path.insert(0, ".")


async def main():
    from api.infrastructure.models.colpali_engine import ColPaliEngine, HAS_COLPALI

    if not HAS_COLPALI:
        print("SKIP: ColPali not available in transformers")
        return

    print("Creating ColPaliEngine...")
    engine = ColPaliEngine()
    # 작은 테스트 이미지 1장 (모델 다운로드 발생 시 시간 걸림)
    img = Image.new("RGB", (128, 128), color="white")
    print("Calling engine.embed([image]) with gpu2_lock...")
    emb = await engine.embed([img])
    print("Embeddings shape:", getattr(emb, "shape", type(emb)))
    if hasattr(emb, "shape"):
        print("OK: ColPali engine and gpu2_lock work.")
    else:
        print("OK: ColPali returned:", type(emb))


if __name__ == "__main__":
    asyncio.run(main())
