import os
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from document_processor import DocumentProcessor
from vector_store import VectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

import logging

# 로깅 설정
logger = logging.getLogger(__name__)

class RAGSystem:
    """RAG(Retrieval-Augmented Generation) 시스템 메인 클래스"""
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 50,
                 db_path: str = "./vector_db"):
        """
        RAG 시스템 초기화
        
        Args:
            chunk_size: 청크 크기 (토큰 수)
            chunk_overlap: 청크 간 겹치는 토큰 수
            db_path: 벡터 DB 저장 경로
        """
        self.document_processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.vector_store = VectorStore(db_path)
        self.llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,
            max_output_tokens=1000
        )
        logger.info("RAG 시스템 초기화 완료")
    
    def add_document(self, file_path: str) -> Dict[str, Any]:
        """
        문서를 벡터 DB에 추가
        """
        try:
            # 1. 텍스트 추출
            logger.info(f"문서 텍스트 추출 시작: {file_path}")
            text = self.document_processor.extract_text(file_path)
            logger.info(f"추출된 텍스트 길이: {len(text)}자")
            
            # 텍스트 검증
            if not text or len(text.strip()) < 10:
                return {
                    "status": "error",
                    "error": "텍스트 추출 실패 또는 텍스트가 너무 짧습니다.",
                    "file_path": file_path
                }
            
            # 2. 메타데이터 생성
            metadata = {
                "source": file_path,
                "file_type": file_path.split('.')[-1].lower(),
                "text_length": len(text)
            }
            
            # 3. 텍스트 청킹
            logger.info("텍스트 청킹 시작")
            documents = self.document_processor.split_text(text, metadata)
            logger.info(f"생성된 청크 개수: {len(documents)}")
            
            # 청크 내용 확인 (디버깅용)
            for i, doc in enumerate(documents[:3]):  # 처음 3개 청크만
                logger.info(f"청크 {i+1}: {doc.page_content[:100]}...")
            
            # 4. 임베딩 생성
            logger.info("임베딩 생성 시작")
            texts = [doc.page_content for doc in documents]
            embeddings = self.document_processor.get_embeddings(texts)
            logger.info(f"생성된 임베딩 개수: {len(embeddings)}")
            
            # 5. 벡터 DB에 저장
            logger.info("벡터 DB에 저장 시작")
            self.vector_store.add_documents(documents, embeddings)
            
            # 저장 후 상태 확인
            stats = self.vector_store.get_stats()
            logger.info(f"저장 후 벡터DB 상태: 총 문서 {stats['total_documents']}개, 인덱스 크기 {stats['index_size']}")
            
            result = {
                "status": "success",
                "file_path": file_path,
                "chunks_created": len(documents),
                "total_tokens": sum(len(text.split()) for text in texts),
                "vector_db_stats": stats
            }
            
            logger.info(f"문서 추가 완료: {result}")
            return result
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            return {
                "status": "error",
                "error": str(e),
                "file_path": file_path
            }
    
    def query(self, question: str, k: int = None) -> Dict[str, Any]:
        """
        질문에 대한 답변 생성
        """
        try:
            # 벡터DB 상태 확인
            stats = self.vector_store.get_stats()
            logger.info(f"벡터DB 상태: 총 문서 {stats['total_documents']}개, 인덱스 크기 {stats['index_size']}")
            
            if stats['total_documents'] == 0:
                logger.warning("벡터DB에 문서가 없습니다. 일반 챗봇 모드로 동작합니다.")
                prompt = question
                documents, metadata_list, scores = [], [], []
            else:
                # 1. 질문 임베딩 생성
                logger.info(f"질문 임베딩 생성: {question}")
                question_embedding = self.document_processor.get_embeddings([question])[0]
                
                # 2. 관련 문서 검색
                logger.info("관련 문서 검색 시작")
                if k is None:
                    k = stats['total_documents']
                documents, metadata_list, scores = self.vector_store.search(question_embedding, k)
                logger.info(f"검색된 문서 개수: {len(documents)}")
                
                # 검색된 문서 내용 로깅 (디버깅용)
                for i, doc in enumerate(documents):
                    #logger.info(f"검색된 문서 {i+1}: {doc[:200]}...")
                
                # 3. 벡터 DB가 비어있으면 context 없이 LLM에게 질문만 전달
                if not documents:
                    logger.info("관련 문서를 찾을 수 없으므로, LLM에게 질문만 전달합니다.")
                    prompt = question
                else:
                    # 4. 컨텍스트 구성
                    context = "\n\n".join(documents)
                    prompt = f"""다음 문서들을 참고하여 질문에 답변해주세요.\n\n문서 내용:\n{context}\n\n질문: {question}\n\n답변:"""
                    logger.info(f"컨텍스트 길이: {len(context)}자")
            
            # 5. LLM으로 답변 생성
            logger.info("LLM으로 답변 생성 시작")
            logger.info(f"프롬프트 길이: {len(prompt)}자")
            response = self.llm.invoke(prompt)
            answer = response.content
            
            result = {
                "status": "success",
                "question": question,
                "answer": answer,
                "context_documents": documents,
                "metadata": metadata_list,
                "similarity_scores": scores,
                "context_length": len(prompt),
                "vector_db_stats": stats
            }
            
            logger.info(f"질문 답변 완료: {len(answer)}자")
            return result
            
        except Exception as e:
            logger.error(f"질문 답변 실패: {e}")
            return {
                "status": "error",
                "error": str(e),
                "question": question
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """시스템 통계 정보 반환"""
        vector_stats = self.vector_store.get_stats()
        return {
            "vector_store": vector_stats,
            "chunk_size": self.document_processor.chunk_size,
            "chunk_overlap": self.document_processor.chunk_overlap
        }
    
    def clear_database(self):
        """벡터 DB 초기화"""
        self.vector_store.clear()
        logger.info("벡터 DB 초기화 완료") 