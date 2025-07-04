import os
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Tuple
from langchain.schema import Document
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

class VectorStore:
    """FAISS 벡터 데이터베이스 관리 클래스"""
    
    def __init__(self, db_path: str = "./vector_db"):
        """
        VectorStore 초기화
        
        Args:
            db_path: 벡터 DB 저장 경로
        """
        self.db_path = db_path
        self.index = None
        self.documents = []
        self.metadata = []
        
        # DB 디렉토리 생성
        os.makedirs(db_path, exist_ok=True)
        
        # 기존 DB 로드 시도
        self._load_existing_db()
    
    def _load_existing_db(self):
        """기존 벡터 DB 로드"""
        index_path = os.path.join(self.db_path, "faiss_index.bin")
        docs_path = os.path.join(self.db_path, "documents.pkl")
        metadata_path = os.path.join(self.db_path, "metadata.pkl")
        
        if (os.path.exists(index_path) and 
            os.path.exists(docs_path) and 
            os.path.exists(metadata_path)):
            try:
                self.index = faiss.read_index(index_path)
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"기존 벡터 DB 로드 완료: {len(self.documents)}개 문서")
            except Exception as e:
                logger.error(f"기존 벡터 DB 로드 실패: {e}")
                self._initialize_new_index()
        else:
            self._initialize_new_index()
    
    def _initialize_new_index(self):
        """새로운 FAISS 인덱스 초기화"""
        # 768은 Gemini 임베딩 차원
        dimension = 768
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
        logger.info(f"새로운 FAISS 인덱스 초기화 (차원: {dimension})")
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """
        문서와 임베딩을 벡터 DB에 추가
        
        Args:
            documents: Document 객체 리스트
            embeddings: 임베딩 벡터 리스트
        """
        try:
            # 임베딩을 numpy 배열로 변환
            embeddings_array = np.array(embeddings, dtype=np.float32)
            
            # FAISS 인덱스에 추가
            self.index.add(embeddings_array)
            
            # 문서와 메타데이터 저장
            for doc in documents:
                self.documents.append(doc.page_content)
                self.metadata.append(doc.metadata)
            
            logger.info(f"벡터 DB에 {len(documents)}개 문서 추가 완료")
            
            # DB 저장
            self._save_db()
            
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            raise
    
    def search(self, query_embedding: List[float], k: int = 5) -> Tuple[List[str], List[Dict[str, Any]], List[float]]:
        """
        쿼리 임베딩과 유사한 문서 검색
        
        Args:
            query_embedding: 쿼리 임베딩 벡터
            k: 반환할 문서 수
            
        Returns:
            (문서 내용 리스트, 메타데이터 리스트, 유사도 점수 리스트)
        """
        try:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("FAISS 인덱스가 비어 있습니다. 문서를 먼저 추가하세요.")
                return [], [], []
            # 쿼리 임베딩을 numpy 배열로 변환
            query_array = np.array([query_embedding], dtype=np.float32)
            
            # FAISS 검색
            scores, indices = self.index.search(query_array, k)
            
            # 결과 추출
            documents = []
            metadata_list = []
            similarity_scores = []
            
            for i, score in zip(indices[0], scores[0]):
                if i < len(self.documents):  # 유효한 인덱스인지 확인
                    documents.append(self.documents[i])
                    metadata_list.append(self.metadata[i])
                    similarity_scores.append(float(score))
            
            logger.info(f"검색 완료: {len(documents)}개 문서 반환")
            return documents, metadata_list, similarity_scores
            
        except Exception as e:
            logger.error(f"검색 실패: {e}")
            raise
    
    def _save_db(self):
        """벡터 DB를 파일로 저장"""
        try:
            # FAISS 인덱스 저장
            index_path = os.path.join(self.db_path, "faiss_index.bin")
            faiss.write_index(self.index, index_path)
            
            # 문서 저장
            docs_path = os.path.join(self.db_path, "documents.pkl")
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # 메타데이터 저장
            metadata_path = os.path.join(self.db_path, "metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
            
            logger.info("벡터 DB 저장 완료")
            
        except Exception as e:
            logger.error(f"벡터 DB 저장 실패: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """벡터 DB 통계 정보 반환"""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "db_path": self.db_path
        }
    
    def clear(self):
        """벡터 DB 초기화"""
        self._initialize_new_index()
        self.documents = []
        self.metadata = []
        self._save_db()
        logger.info("벡터 DB 초기화 완료") 