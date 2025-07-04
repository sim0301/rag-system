import os
import pdfplumber
import tiktoken
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """문서 처리, 청킹, 임베딩을 담당하는 클래스"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        DocumentProcessor 초기화
        
        Args:
            chunk_size: 청크 크기 (토큰 수)
            chunk_overlap: 청크 간 겹치는 토큰 수
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        #self.embeddings = OpenAIEmbeddings()
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def _count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수를 계산"""
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        PDF 파일에서 텍스트 추출 (pdfplumber 사용) - 개선된 버전
        
        Args:
            file_path: PDF 파일 경로
            
        Returns:
            추출된 텍스트
        """
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                logger.info(f"PDF 페이지 수: {len(pdf.pages)}")
                
                for i, page in enumerate(pdf.pages):
                    try:
                        # 더 상세한 텍스트 추출 옵션
                        page_text = page.extract_text(
                            layout=True,  # 레이아웃 정보 포함
                            x_tolerance=3,  # x축 허용 오차
                            y_tolerance=3   # y축 허용 오차
                        )
                        
                        if page_text:
                            text += f"\n--- Page {i+1} ---\n{page_text}\n"
                            logger.info(f"Page {i+1} 텍스트 길이: {len(page_text)}")
                        else:
                            logger.warning(f"Page {i+1}에서 텍스트 추출 실패")
                            
                    except Exception as e:
                        logger.error(f"Page {i+1} 처리 중 오류: {e}")
                        continue
            
            logger.info(f"PDF 텍스트 추출 완료: 총 {len(text)}자")
            return text
            
        except Exception as e:
            logger.error(f"PDF 텍스트 추출 실패: {e}")
            raise
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """
        TXT 파일에서 텍스트 추출
        
        Args:
            file_path: TXT 파일 경로
            
        Returns:
            추출된 텍스트
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            logger.info(f"TXT 텍스트 추출 완료: {file_path}")
            return text
        except Exception as e:
            logger.error(f"TXT 텍스트 추출 실패: {e}")
            raise
    
    def extract_text(self, file_path: str) -> str:
        """
        파일 확장자에 따라 적절한 텍스트 추출 메서드 호출
        
        Args:
            file_path: 파일 경로
            
        Returns:
            추출된 텍스트
        """
        file_extension = file_path.lower().split('.')[-1]
        
        if file_extension == 'pdf':
            text = self.extract_text_from_pdf(file_path)
            # 추출된 텍스트 앞부분 로그 출력
            logger.info(f"추출된 텍스트 앞 500자: {text[:500]}")
            return text
        elif file_extension == 'txt':
            return self.extract_text_from_txt(file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {file_extension}")
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """
        텍스트를 청크로 분할
        
        Args:
            text: 분할할 텍스트
            metadata: 문서 메타데이터
            
        Returns:
            Document 객체 리스트
        """
        try:
            documents = self.text_splitter.create_documents(
                texts=[text],
                metadatas=[metadata or {}]
            )
            logger.info(f"텍스트 청킹 완료: {len(documents)}개 청크 생성")
            return documents
        except Exception as e:
            logger.error(f"텍스트 청킹 실패: {e}")
            raise
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        텍스트 리스트를 임베딩 벡터로 변환
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            임베딩 벡터 리스트
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"임베딩 완료: {len(texts)}개 텍스트")
            return embeddings
        except Exception as e:
            logger.error(f"임베딩 실패: {e}")
            raise 