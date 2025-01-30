import os

import streamlit as st
from github import Github
from github.GithubException import UnknownObjectException
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 환경 변수 로드
load_dotenv()

# GitHub 및 OpenAI 클라이언트 초기화
github_client = Github(os.getenv("GITHUB_TOKEN"))
llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
embeddings = OpenAIEmbeddings()
# llm = ChatOpenAI(model="gpt-4")


def is_valid_github_url(url):
    """GitHub URL이 유효한지 확인"""
    if not url:
        return False
    if not ("github.com" in url):
        return False
    try:
        # URL에서 owner와 repo 추출
        parts = url.strip("/").split("/")
        owner = parts[-2]
        repo = parts[-1]
        # 레포지토리 존재 여부 확인
        github_client.get_repo(f"{owner}/{repo}")
        return True
    except (IndexError, UnknownObjectException):
        return False


def get_repo_contents(url):
    """레포지토리의 주요 코드 파일들과 README를 가져옴"""
    parts = url.strip("/").split("/")
    owner = parts[-2]
    repo = parts[-1]

    repo = github_client.get_repo(f"{owner}/{repo}")
    contents = []
    readme_content = ""

    def process_contents(repo_contents):
        for content in repo_contents:
            if content.type == "dir":
                process_contents(repo.get_contents(content.path))
            elif content.type == "file":
                if content.name.lower() == "readme.md":
                    readme_content = content.decoded_content.decode("utf-8")
                elif content.path.endswith(
                    (".py", ".js", ".java", ".cpp", ".ts", ".dart", ".go", ".rs")
                ):
                    contents.append(
                        {
                            "path": content.path,
                            "content": content.decoded_content.decode("utf-8"),
                            "url": content.html_url,  # GitHub 파일 링크 추가
                        }
                    )

    process_contents(repo.get_contents(""))
    return contents, readme_content


def analyze_code(contents, readme_content):
    """GPT를 사용하여 코드와 README 분석"""
    system_message = SystemMessage(
        content="""
        당신은 전문적인 코드 리뷰어입니다. README 내용과 코드를 분석하여 다음 항목들을 평가해주세요:

        1. 프로젝트 개요:
        - README를 기반으로 한 프로젝트의 목적과 주요 기능
        
        2. 상세 평가 (각 항목별 점수와 근거 제시):

        A. 코드 품질 (40점)
        - 코드 구조 및 설계 (10점)
        - 가독성 및 네이밍 (10점)
        - 모듈화/재사용성 (10점)
        - 에러 처리 (10점)

        B. 문서화 (20점)
        - README 완성도 (10점)
        - 코드 주석 (10점)

        C. 보안 및 성능 (20점)
        - 보안 취약점 (10점)
        - 성능 최적화 (10점)

        D. 프로젝트 완성도 (20점)
        - 기능 구현 완성도 (10점)
        - 테스트 커버리지 (10점)

        3. 총점 및 등급:
        - 총점 (100점 만점)
        - 등급 (A+: 95-100, A: 90-94, B+: 85-89, B: 80-84, C+: 75-79, C: 70-74, D: 60-69, F: 0-59)
        
        4. 개선 제안:
        - 각 항목별 구체적인 개선 방안
        - 관련 파일의 GitHub 링크 포함

        점수 산정 시 각 항목별로 구체적인 근거를 제시하고, 개선이 필요한 부분에 대해 실질적인 해결 방안을 제시해주세요.
        """
    )

    # README와 코드 내용을 함께 전달
    content_with_links = {"readme": readme_content, "files": contents}

    human_message = HumanMessage(content=str(content_with_links))
    messages = [system_message, human_message]

    response = llm.invoke(messages)
    return response.content


def create_vector_db(contents, readme_content):
    """코드와 README를 벡터 DB에 저장"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )

    documents = []

    # README 처리
    if readme_content:
        readme_chunks = text_splitter.split_text(readme_content)
        for chunk in readme_chunks:
            documents.append(f"[README.md] {chunk}")

    # 코드 파일 처리
    for file in contents:
        code_chunks = text_splitter.split_text(file["content"])
        for chunk in code_chunks:
            documents.append(f"[{file['path']}] {chunk}")

    # Chroma DB 생성
    vectordb = Chroma.from_texts(
        texts=documents, embedding=embeddings, persist_directory="./chroma_db"
    )

    return vectordb


def setup_qa_chain(vectordb):
    """QA 체인 설정"""
    # 컨텍스트를 고려한 질문 생성을 위한 프롬프트
    contextualize_q_system_prompt = """
    주어진 대화 기록과 최신 질문을 바탕으로, 
    대화 기록 없이도 이해할 수 있는 독립적인 질문을 만드세요.
    질문에 답하지 말고, 필요한 경우 질문을 재구성하거나 그대로 반환하세요.
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 컨텍스트 인식 검색기 생성
    retriever = create_history_aware_retriever(
        llm, vectordb.as_retriever(), contextualize_q_prompt
    )

    # 질문 응답을 위한 프롬프트
    qa_system_prompt = """
    당신은 코드 리뷰 전문가입니다. 
    주어진 컨텍스트를 사용하여 질문에 답변하세요. 
    답을 모르는 경우 모른다고 말씀하세요. 
    최대 세 문장으로 간단명료하게 답변하세요.
    
    컨텍스트:
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 문서 결합 체인 생성
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 최종 검색 체인 생성
    qa_chain = create_retrieval_chain(retriever, question_answer_chain)

    return qa_chain


# Streamlit UI
st.title("GitHub 레포지토리 코드 리뷰 챗봇")

# GitHub URL 입력
repo_url = st.text_input("GitHub 레포지토리 URL을 입력하세요:")

if repo_url:
    if not is_valid_github_url(repo_url):
        st.error("유효하지 않은 GitHub 레포지토리 URL입니다.")
    else:
        with st.spinner("레포지토리 분석 중..."):
            try:
                # 코드와 README 가져오기
                contents, readme_content = get_repo_contents(repo_url)

                # 초기 코드 분석 수행
                if "initial_analysis" not in st.session_state:
                    analysis = analyze_code(contents, readme_content)
                    st.session_state.initial_analysis = analysis

                    # 벡터 DB 생성 및 QA 체인 설정
                    vectordb = create_vector_db(contents, readme_content)
                    st.session_state.qa_chain = setup_qa_chain(vectordb)

                # 초기 분석 결과 표시
                st.markdown("## 초기 코드 리뷰 결과")
                st.markdown(st.session_state.initial_analysis)

                # 추가 질의응답 섹션
                st.markdown("---")
                st.markdown("## 추가 질문하기")
                st.markdown("코드에 대해 더 자세히 알고 싶은 점을 질문해주세요.")

                # 채팅 기록 초기화
                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []

                question = st.text_input("질문을 입력하세요:", key="question_input")

                if question:
                    with st.spinner("답변 생성 중..."):
                        result = st.session_state.qa_chain.invoke(
                            {
                                "input": question,
                                "chat_history": st.session_state.chat_history,
                            }
                        )

                        # 채팅 기록 업데이트
                        st.session_state.chat_history.extend(
                            [("human", question), ("assistant", result["answer"])]
                        )

                        st.markdown("### 답변:")
                        st.markdown(result["answer"])

                        # 참조된 소스 코드 표시
                        with st.expander("참조된 소스 코드 보기"):
                            for doc in result["context"]:
                                st.code(doc.page_content)

            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")

# 사용 방법 안내
with st.expander("사용 방법"):
    st.markdown(
        """
    1. GitHub 레포지토리 URL을 입력창에 붙여넣으세요.
    2. 자동으로 코드 분석이 시작됩니다.
    3. 분석이 완료되면 코드의 장단점과 개선 방안을 확인할 수 있습니다.
    
    **참고**: 
    - 큰 레포지토리의 경우 분석에 시간이 걸릴 수 있습니다.
    - 현재는 Python, JavaScript, Java, C++, TypeScript, Dart, Go, Rust 파일만 분석합니다.
    """
    )
