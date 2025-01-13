# ALL-in 코딩 공모전 수상작 RAG 챗봇 프로젝트

## 프로젝트 소개
이 프로젝트는 LangChain을 활용하여 외부 블로그의 정보를 기반으로 RAG(Retrieval-Augmented Generation) 챗봇을 구현한 것입니다. 특정 블로그 글에서 정보를 검색하고 관련 질문에 답변할 수 있는 시스템을 개발했습니다.

## 주요 기능
- 스파르타코딩클럽 블로그의 ALL-in 코딩 공모전 수상작 정보 검색
- RAG 기반 질의응답 시스템
- GPT-4를 활용한 자연어 응답 생성

## RAG(Retrieval-Augmented Generation)란?
RAG는 대규모 언어 모델(LLM)의 성능을 향상시키기 위해 외부 지식을 검색하여 활용하는 방식입니다. 기본 작동 원리는 다음과 같습니다:

1. 검색(Retrieval): 사용자의 질문과 관련된 문서나 정보를 외부 데이터베이스에서 검색
2. 증강(Augmentation): 검색된 정보를 프롬프트에 추가
3. 생성(Generation): LLM이 증강된 컨텍스트를 기반으로 답변 생성

RAG의 장점:
- 최신 정보 반영 가능
- 사실 기반의 응답 생성
- 환각(Hallucination) 감소
- 특정 도메인에 특화된 응답 가능

## LangChain
LangChain은 LLM 애플리케이션 개발을 위한 프레임워크입니다. 이 프로젝트에서는 다음과 같은 LangChain의 주요 컴포넌트를 활용했습니다:

- WebBaseLoader: 웹 페이지 컨텐츠 로딩
- RecursiveCharacterTextSplitter: 문서 청크 분할
- Chroma: 벡터 데이터베이스
- OpenAIEmbeddings: 텍스트 임베딩
- ChatOpenAI: LLM 인터페이스

## 설치 방법
```bash
pip install langchain-community langchain-chroma langchain-openai bs4
```

## 환경 설정
프로젝트 실행을 위해 다음 환경 변수 설정이 필요합니다:
```python
# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

# LangSmith 설정 (선택사항)
os.environ["LANGSMITH_API_KEY"] = "your_langsmith_api_key"
os.environ["LANGSMITH_PROJECT"] = "project_name"
```

## 사용 예시
```python
# RAG 시스템에 질문하기
user_msg = "ALL-in 코딩 공모전 수상작들을 요약해줘."
retrieved_docs = retriever.invoke(user_msg)
prompt = hub.pull("rlm/rag-prompt")
response = llm.invoke(prompt.invoke({
    "context": format_docs(retrieved_docs), 
    "question": user_msg
}))
```

## 기술 스택
- Python
- LangChain
- OpenAI GPT-4
- ChromaDB
- BeautifulSoup4

## 주의사항
- 한글 인코딩을 위해 UTF-8 설정이 필요합니다
- 웹 크롤링 시 해당 웹사이트의 robots.txt를 확인하세요
- API 키는 보안을 위해 환경변수로 관리하세요

## License
This project is licensed under the MIT License - see the LICENSE file for details
