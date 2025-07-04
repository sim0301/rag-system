import os
import sys
from dotenv import load_dotenv
from rag_system import RAGSystem
import logging

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """RAG 시스템 메인 실행 함수"""
    
    # OpenAI API 키 확인
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        logger.info("env_example.txt 파일을 참고하여 .env 파일을 생성하세요.")
        return
    
    # RAG 시스템 초기화
    rag = RAGSystem(
        chunk_size=int(os.getenv('CHUNK_SIZE', 500)),
        chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 50)),
        db_path=os.getenv('VECTOR_DB_PATH', './vector_db')
    )
    
    print("=== RAG 시스템 시작 ===")
    print("1. 문서 추가")
    print("2. 질문하기")
    print("3. 시스템 통계")
    print("4. DB 초기화")
    print("5. 종료")
    
    while True:
        try:
            choice = input("\n선택하세요 (1-5): ").strip()
            
            if choice == '1':
                # 문서 추가
                file_path = input("문서 파일 경로를 입력하세요: ").strip()
                if os.path.exists(file_path):
                    result = rag.add_document(file_path)
                    if result['status'] == 'success':
                        print(f"✅ 문서 추가 성공!")
                        print(f"   - 생성된 청크: {result['chunks_created']}개")
                        print(f"   - 총 토큰 수: {result['total_tokens']}")
                    else:
                        print(f"❌ 문서 추가 실패: {result['error']}")
                else:
                    print("❌ 파일이 존재하지 않습니다.")
            
            elif choice == '2':
                # 질문하기
                question = input("질문을 입력하세요: ").strip()
                if question:
                    print("🤔 답변을 생성 중입니다...")
                    result = rag.query(question)
                    if result['status'] == 'success':
                        print(f"\n💡 답변:")
                        print(result['answer'])
                        print(f"\n📊 관련 문서 수: {len(result['context_documents'])}개")
                        print(f"📏 컨텍스트 길이: {result['context_length']}자")
                    else:
                        print(f"❌ 답변 생성 실패: {result['error']}")
                else:
                    print("❌ 질문을 입력해주세요.")
            
            elif choice == '3':
                # 시스템 통계
                stats = rag.get_stats()
                print("\n📈 시스템 통계:")
                print(f"   - 총 문서 수: {stats['vector_store']['total_documents']}")
                print(f"   - 인덱스 크기: {stats['vector_store']['index_size']}")
                print(f"   - 청크 크기: {stats['chunk_size']}")
                print(f"   - 청크 겹침: {stats['chunk_overlap']}")
                print(f"   - DB 경로: {stats['vector_store']['db_path']}")
            
            elif choice == '4':
                # DB 초기화
                confirm = input("정말로 벡터 DB를 초기화하시겠습니까? (y/N): ").strip().lower()
                if confirm == 'y':
                    rag.clear_database()
                    print("✅ 벡터 DB 초기화 완료")
                else:
                    print("❌ 초기화가 취소되었습니다.")
            
            elif choice == '5':
                # 종료
                print("👋 RAG 시스템을 종료합니다.")
                break
            
            else:
                print("❌ 잘못된 선택입니다. 1-5 중에서 선택해주세요.")
                
        except KeyboardInterrupt:
            print("\n👋 RAG 시스템을 종료합니다.")
            break
        except Exception as e:
            logger.error(f"오류 발생: {e}")
            print(f"❌ 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main() 