from vector_store import VectorStore
from document_processor import DocumentProcessor
#dp = DocumentProcessor(chunk_size=1000)

# 벡터DB 경로는 실제 사용 경로로 맞춰주세요
vs = VectorStore(db_path="./vector_db")

print("=== 저장된 문서 청크 ===")
for i, doc in enumerate(vs.documents):
    if i == 0:
        print(f"[{i}] {doc[:10000]}...")  # 앞 100자만 출력

#print("\n=== 메타데이터 ===")
#for i, meta in enumerate(vs.metadata):
#    print(f"[{i}] {meta}")


#text = dp.extract_text(r"vs.documents")
#print("1조" in text)  # True/False