# LLM-based Fashion Recommendation System

## Overview
H&M 데이터셋 기반 패션 추천 시스템입니다. 자연어 쿼리와 유저 히스토리를 기반으로 개인화된 패션 아이템을 추천합니다.

## Architecture
User Query / User History  
→ ChromaDB Retrieval (후보 20개)  
→ 중복 제거 (후보 5개)  
→ Qwen3-VL-2B-Instruct Reranking  
→ 추천 결과 + 한국어 이유  

## Dataset
- H&M Personalized Fashion Recommendations (Kaggle)
- 105,542 items / 1,371,980 users / 31,788,324 transactions

## Tech Stack
- Python, PyTorch
- Embedding: Qwen3-VL-Embedding-2B (multimodal)
- Vector Search: ChromaDB
- LLM: Qwen3-VL-2B-Instruct (local)

## Roadmap
- [x] v1: 자연어 입력 → FAISS retrieval → Qwen2.5 reranking + 한국어 추천 이유 파인튜닝 (텍스트 전용)
- [x] v1: 유저 구매 이력 임베딩 평균 기반 개인화 추천
- [x] v2: 멀티모달 전환 (Qwen3-VL-Embedding-2B + ChromaDB + 이미지 추가)
- [ ] v3: 유저 로그인 + 협업 필터링
- [ ] v4: 날씨/일정/장소 기반 오늘의 옷 추천
