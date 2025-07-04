# RAG (Retrieval-Augmented Generation) 시스템

ChatGPT API를 사용한 문서 기반 질의응답 시스템입니다.

## 🚀 주요 기능

- **문서 업로드**: PDF, TXT 파일 지원
- **텍스트 청킹**: 500 토큰 단위로 문서 분할
- **벡터 임베딩**: OpenAI Embedding API 사용
- **FAISS 벡터 DB**: 고속 유사도 검색
- **GPT-4 답변 생성**: 검색된 문서 기반 답변
- **React 웹 인터페이스**: 직관적인 사용자 경험

## 🏗️ 시스템 구성

### 백엔드 (Python)
- `document_processor.py`: 문서 처리, 청킹, 임베딩
- `vector_store.py`: FAISS 벡터 데이터베이스 관리
- `rag_system.py`: RAG 시스템 메인 클래스
- `main.py`: CLI 인터페이스
- `flask_app.py`: Flask 웹 서버

### 프론트엔드 (React)
- 문서 업로드 인터페이스
- 질문 입력 및 답변 표시
- 시스템 통계 대시보드
- 벡터 DB 관리 기능

## 📦 설치 및 실행

### 1. 환경 설정
```bash
# 프로젝트 클론
git clone <repository-url>
cd rag-system

# Python 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경변수 설정
```bash
# .env 파일 생성
cp env_example.txt .env

# .env 파일 편집
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_API_BASE=https://api.openai.com/v1
VECTOR_DB_PATH=./vector_db
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### 3. 백엔드 실행
```bash
# CLI 모드
python main.py

# 또는 Flask 웹 서버
python flask_app.py
```

### 4. 프론트엔드 실행
```bash
cd frontend
npm install
npm start
```

## 🔧 사용 방법

### CLI 모드
1. `python main.py` 실행
2. 메뉴에서 선택:
   - 1: 문서 추가
   - 2: 질문하기
   - 3: 시스템 통계
   - 4: DB 초기화
   - 5: 종료

### 웹 인터페이스
1. Flask 서버 실행: `python flask_app.py`
2. React 앱 실행: `cd frontend && npm start`
3. 브라우저에서 `http://localhost:3000` 접속
4. 문서 업로드 후 질문 입력

## 📁 프로젝트 구조
```
rag-system/
├── requirements.txt          # Python 의존성
├── env_example.txt          # 환경변수 예시
├── document_processor.py    # 문서 처리 모듈
├── vector_store.py         # FAISS 벡터 DB
├── rag_system.py           # RAG 시스템 메인
├── main.py                 # CLI 인터페이스
├── flask_app.py            # Flask 웹 서버
├── frontend/               # React 앱
│   ├── package.json
│   ├── public/
│   └── src/
│       ├── App.js
│       ├── App.css
│       └── index.js
└── README.md
```

## 🔍 API 엔드포인트

### Flask 서버 API
- `POST /api/upload`: 문서 업로드
- `POST /api/query`: 질문 처리
- `GET /api/stats`: 시스템 통계
- `POST /api/clear`: 벡터 DB 초기화
- `GET /api/health`: 헬스 체크

## ⚙️ 설정 옵션

### 청킹 설정
- `CHUNK_SIZE`: 청크 크기 (기본값: 500 토큰)
- `CHUNK_OVERLAP`: 청크 간 겹침 (기본값: 50 토큰)

### 벡터 DB 설정
- `VECTOR_DB_PATH`: 벡터 DB 저장 경로 (기본값: ./vector_db)

### OpenAI 설정
- `OPENAI_API_KEY`: OpenAI API 키
- `OPENAI_API_BASE`: OpenAI API 베이스 URL

## 🛠️ 개발 정보

### 기술 스택
- **백엔드**: Python, Flask, LangChain, FAISS
- **프론트엔드**: React, Axios
- **AI/ML**: OpenAI GPT-4, OpenAI Embeddings
- **데이터베이스**: FAISS 벡터 DB

### 주요 라이브러리
- `langchain`: LLM 프레임워크
- `faiss-cpu`: 벡터 유사도 검색
- `PyPDF2`: PDF 텍스트 추출
- `tiktoken`: 토큰 카운팅
- `flask-cors`: CORS 지원

## 📝 라이선스

MIT License

## 🤝 기여

이슈 및 풀 리퀘스트 환영합니다! 