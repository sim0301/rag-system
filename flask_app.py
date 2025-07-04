import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from rag_system import RAGSystem
import logging
from langchain_google_genai import ChatGoogleGenerativeAI

# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # React 앱과의 통신을 위해 CORS 활성화

# 업로드 폴더 설정
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'hwp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# RAG 시스템 초기화
rag = RAGSystem(
    chunk_size=int(os.getenv('CHUNK_SIZE', 1000)),
    chunk_overlap=int(os.getenv('CHUNK_OVERLAP', 50)),
    db_path=os.getenv('VECTOR_DB_PATH', './vector_db')
)

llm = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def allowed_file(filename):
    """허용된 파일 확장자 확인"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """문서 업로드 및 벡터 DB에 추가"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '파일이 없습니다.'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '선택된 파일이 없습니다.'}), 400
        
        if file and allowed_file(file.filename):
            # 파일 저장
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # RAG 시스템에 문서 추가
            result = rag.add_document(filepath)
            
            if result['status'] == 'success':
                return jsonify({
                    'message': '문서가 성공적으로 추가되었습니다.',
                    'chunks_created': result['chunks_created'],
                    'total_tokens': result['total_tokens']
                })
            else:
                return jsonify({'error': result['error']}), 500
        
        return jsonify({'error': '지원하지 않는 파일 형식입니다.'}), 400
        
    except Exception as e:
        logger.error(f"문서 업로드 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query():
    """질문에 대한 답변 생성"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': '질문을 입력해주세요.'}), 400
        
        # RAG 시스템으로 질문 처리
        result = rag.query(question)
        
        if result['status'] == 'success':
            logger.debug(result['answer'])
            return jsonify({
                'answer': result['answer'],
                'context_documents': result['context_documents'],
                'similarity_scores': result['similarity_scores'],
                'context_length': result['context_length']
            })
        else:
            return jsonify({'error': result['error']}), 500
            
    except Exception as e:
        logger.error(f"질문 처리 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """시스템 통계 정보 반환"""
    try:
        stats = rag.get_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"통계 조회 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear', methods=['POST'])
def clear_database():
    """벡터 DB 초기화"""
    try:
        rag.clear_database()
        return jsonify({'message': '벡터 DB가 초기화되었습니다.'})
    except Exception as e:
        logger.error(f"DB 초기화 오류: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """헬스 체크"""
    return jsonify({'status': 'healthy', 'message': 'RAG 시스템이 정상 작동 중입니다.'})

if __name__ == '__main__':
    # OpenAI API 키 확인
    if not os.getenv('GOOGLE_API_KEY'):
        logger.error("GOOGLE_API_KEY 환경변수가 설정되지 않았습니다.")
        logger.info("env_example.txt 파일을 참고하여 .env 파일을 생성하세요.")
        exit(1)
    
    logger.info("Flask 서버 시작...")
    app.run(debug=True, host='0.0.0.0', port=5000) 