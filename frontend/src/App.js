import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [stats, setStats] = useState(null);

  // 시스템 통계 가져오기
  const fetchStats = async () => {
    try {
      const response = await axios.get('/api/stats');
      setStats(response.data);
    } catch (error) {
      console.error('통계 조회 실패:', error);
    }
  };

  useEffect(() => {
    fetchStats();
  }, []);

  // 파일 업로드 처리
  const handleFileUpload = async () => {
    if (!file) {
      alert('파일을 선택해주세요.');
      return;
    }

    setLoading(true);
    setUploadStatus('업로드 중...');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setUploadStatus(`✅ ${response.data.message} (청크: ${response.data.chunks_created}개)`);
      fetchStats(); // 통계 업데이트
    } catch (error) {
      setUploadStatus(`❌ 업로드 실패: ${error.response?.data?.error || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 질문 처리
  const handleQuestion = async () => {
    if (!question.trim()) {
      alert('질문을 입력해주세요.');
      return;
    }

    setLoading(true);
    setAnswer('답변을 생성 중입니다...');

    try {
      const response = await axios.post('/api/query', {
        question: question.trim(),
      });

      setAnswer(response.data.answer);
    } catch (error) {
      setAnswer(`❌ 오류: ${error.response?.data?.error || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // DB 초기화
  const handleClearDB = async () => {
    if (!window.confirm('정말로 벡터 DB를 초기화하시겠습니까?')) {
      return;
    }

    try {
      await axios.post('/api/clear');
      alert('벡터 DB가 초기화되었습니다.');
      fetchStats(); // 통계 업데이트
    } catch (error) {
      alert(`초기화 실패: ${error.response?.data?.error || error.message}`);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🤖 RAG 시스템</h1>
        <p>문서 기반 질의응답 시스템</p>
      </header>

      <main className="App-main">
        {/* 시스템 통계 */}
        {stats && (
          <div className="stats-section">
            <h3>📊 시스템 통계</h3>
            <div className="stats-grid">
              <div className="stat-item">
                <span className="stat-label">총 문서 수:</span>
                <span className="stat-value">{stats.vector_store?.total_documents || 0}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">인덱스 크기:</span>
                <span className="stat-value">{stats.vector_store?.index_size || 0}</span>
              </div>
              <div className="stat-item">
                <span className="stat-label">청크 크기:</span>
                <span className="stat-value">{stats.chunk_size || 500}</span>
              </div>
            </div>
          </div>
        )}

        {/* 문서 업로드 섹션 */}
        <section className="upload-section">
          <h3>📄 문서 업로드</h3>
          <div className="upload-container">
            <input
              type="file"
              accept=".pdf,.txt"
              onChange={(e) => setFile(e.target.files[0])}
              className="file-input"
            />
            <button 
              onClick={handleFileUpload} 
              disabled={loading || !file}
              className="upload-btn"
            >
              {loading ? '업로드 중...' : '업로드'}
            </button>
          </div>
          {uploadStatus && (
            <div className="upload-status">
              {uploadStatus}
            </div>
          )}
        </section>

        {/* 질문 섹션 */}
        <section className="question-section">
          <h3>❓ 질문하기</h3>
          <div className="question-container">
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="문서에 대해 질문해보세요..."
              className="question-input"
              rows="3"
            />
            <button 
              onClick={handleQuestion} 
              disabled={loading || !question.trim()}
              className="question-btn"
            >
              {loading ? '답변 생성 중...' : '질문하기'}
            </button>
          </div>
        </section>

        {/* 답변 섹션 */}
        {answer && (
          <section className="answer-section">
            <h3>💡 답변</h3>
            <div className="answer-container">
              <pre className="answer-text">{answer}</pre>
            </div>
          </section>
        )}

        {/* 관리 섹션 */}
        <section className="admin-section">
          <h3>⚙️ 관리</h3>
          <button onClick={handleClearDB} className="clear-btn">
            벡터 DB 초기화
          </button>
        </section>
      </main>
    </div>
  );
}

export default App; 