import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
import pdfplumber


load_dotenv()
class Settings():
    GEMA3_API_URL: str = os.getenv("GEMA3_API_URL", "http://10.250.141.23:11434")
    GEMA3_MODEL: str = os.getenv("GEMA3_MODEL", "orieg/gemma3-tools:27b")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M")
    AOAI_API_KEY: str | None = os.getenv("AOAI_API_KEY")
    AOAI_ENDPOINT: str | None = os.getenv("AOAI_ENDPOINT")
    AOAI_DEPLOY_GPT4O: str | None = os.getenv("AOAI_DEPLOY_GPT4O")
    AOAI_EMBEDDING_DEPLOYMENT: str | None = os.getenv("AOAI_DEPLOY_EMBED_3_LARGE")
    AOAI_API_VERSION: str = "2024-10-21"

    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_HOST: str | None = None

    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    DB_PATH: str = "history.db"
    SQLALCHEMY_DATABASE_URI: str = f"sqlite:///./{DB_PATH}"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    def get_llm(self):
        """Azure OpenAI LLM 인스턴스를 반환합니다."""
        # return AzureChatOpenAI(
        #     openai_api_key=self.AOAI_API_KEY,
        #     azure_endpoint=self.AOAI_ENDPOINT,
        #     azure_deployment=self.AOAI_DEPLOY_GPT4O,
        #     api_version=self.AOAI_API_VERSION,
        #     temperature=0.7,
        #     streaming=True,
        # )
        return OllamaLLM(
            base_url=self.GEMA3_API_URL,
            model=self.GEMA3_MODEL,
            temperature=0.7
        )

    def get_embeddings(self):
        """Ollama Embeddings 인스턴스를 반환합니다."""
        return OllamaEmbeddings(
            base_url=self.GEMA3_API_URL,
            model=self.EMBEDDING_MODEL
        )


settings = Settings()


def get_llm():
    return settings.get_llm()


def get_embeddings():
    return settings.get_embeddings()


def extract_tables_from_pdf(file_path):
    table_texts = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                formatted_table = "\n".join([
                    "\t".join([
                        str(cell).replace("\n", " ").strip() if cell is not None else ""
                        for cell in row
                    ]) for row in table
                ])
                table_texts.append(formatted_table)
    return table_texts


def save_vectorstore():
    loader = PyMuPDFLoader("./data.pdf")
    docs = loader.load()

    table_texts = extract_tables_from_pdf("./data.pdf")
    table_docs = [Document(page_content=t, metadata={"source": "table"}) for t in table_texts]

    all_docs = docs + table_docs

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_documents = text_splitter.split_documents(all_docs)
    embeddings = settings.get_embeddings()
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    vectorstore.save_local("./vector_index")

    return vectorstore


def load_vectorstore():
    embeddings = settings.get_embeddings()
    vectorstore = FAISS.load_local(
        "./vector_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore