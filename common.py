# -*- coding: utf-8 -*-
"""
common.py — 모든 실습 공통 파일

목적:
  - .env 의 API 키를 한 곳에서 로드한다.
  - Gemini(주력) / OpenAI(보조) 모델 객체를 일관되게 생성한다.
  - 실습 데이터(data/) 경로를 쉽게 찾는다.

각 강의 실습 코드 맨 위에서 다음처럼 불러 씁니다:
    from common import get_chat, get_openai_client, get_genai_client, DATA
"""
import os
import pathlib
from dotenv import load_dotenv

# 프로젝트 루트 및 데이터 경로
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DOCS = DATA / "docs"

# .env 로드 (프로젝트 루트의 .env 읽음)
load_dotenv(ROOT / ".env")

# 환경변수 기본값 설정
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


def require_key(name: str) -> str:
    """환경변수 키가 없거나 플레이스홀더 상태이면 프로그램을 종료함."""
    val = os.getenv(name)
    if not val or val.startswith("여기에"):
        raise SystemExit(                                                        # raise : 강제 발생
            f"[설정 필요] {name} 가 .env 에 없거나 기본값 상태입니다.\n"
            f" 1) cp .env.example .env\n"
            f" 2) .env 파일을 열어 {name} 값을 채우세요."
        )
    return val


# ---------- raw SDK (원리 학습용) ----------
def get_genai_client():
    """google-genai 공식 클라이언트 (from google import genai)."""
    from google import genai
    return genai.Client(api_key=require_key("GOOGLE_API_KEY"))


def get_openai_client():
    """openai 공식 클라이언트 (from openai import OpenAI)."""
    from openai import OpenAI
    return OpenAI(api_key=require_key("OPENAI_API_KEY"))




#==================================================================
#               프로젝트 시에는 LangChain으로 사용
#==================================================================

# ---------- LangChain Chat 모델 (현업용) ----------
def get_chat(provider: str = "openai", temperature: float = 0.0):
    """
    LangChain ChatModel 반환.
    :param provider: 'openai'(기본값) | 'gemini'
    :param temperature: 창의성 조절 매개변수 (기본 0.0)
    """

    if provider == "openai":
            require_key("OPENAI_API_KEY")
            from langchain_openai import ChatOpenAI
            # 상단 환경변수에 정의된 OPENAI_MODEL 사용
            return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature)     
    elif provider == "gemini":
        require_key("GOOGLE_API_KEY")
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temperature)
    raise ValueError(f"알 수 없는 provider입니다: {provider}. ('openai'  또는 'gemini' 사용)")
   


def get_embeddings(provider: str = "openai"):
    """
    LangChain Embeddings 반환.
    :param provider: 'openai'(기본값) | 'gemini'
    """
    
        
    if provider == "openai":
        require_key("OPENAI_API_KEY")
        from langchain_openai import OpenAIEmbeddings
        # 상단 환경변수에 정의된 OPENAI_EMBED_MODEL 사용
        return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    elif provider == "gemini":
        require_key("GOOGLE_API_KEY")
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        # gemini-embedding-001 기본 출력은 3072차원. 저장 및 속도를 위해 768로 설정
        return GoogleGenerativeAIEmbeddings(
            model=GEMINI_EMBED_MODEL, 
            output_dimensionality=768
            )
        
    raise ValueError(f"알 수 없는 provider입니다: {provider}. ('gemini' 또는 'openai' 사용)")


if __name__ == "__main__":
    print("=== [common.py] 환경 점검 ===")
    print("ROOT 경로 :", ROOT)
    print("DATA 경로 :", DATA, f"(존재 여부: {DATA.exists()})")
    print("GEMINI_MODEL :", GEMINI_MODEL)
    print("OPENAI_MODEL :", OPENAI_MODEL)
    print("키 로드 상태:")
    print(" - GOOGLE_API_KEY:", bool(os.getenv("GOOGLE_API_KEY")))
    print(" - OPENAI_API_KEY:", bool(os.getenv("OPENAI_API_KEY")))