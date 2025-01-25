import os

import streamlit as st
from github import Github
from github.GithubException import UnknownObjectException
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# 환경 변수 로드
load_dotenv()

# GitHub 및 OpenAI 클라이언트 초기화
github_client = Github(os.getenv("GITHUB_TOKEN"))
llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
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
    """레포지토리의 주요 코드 파일들을 가져옴"""
    parts = url.strip("/").split("/")
    owner = parts[-2]
    repo = parts[-1]

    repo = github_client.get_repo(f"{owner}/{repo}")
    contents = []

    def process_contents(repo_contents):
        for content in repo_contents:
            if content.type == "dir":
                process_contents(repo.get_contents(content.path))
            elif content.type == "file" and content.path.endswith(
                (".py", ".js", ".java", ".cpp", ".ts")
            ):
                contents.append(
                    {
                        "path": content.path,
                        "content": content.decoded_content.decode("utf-8"),
                    }
                )

    process_contents(repo.get_contents(""))
    return contents


def analyze_code(contents):
    """GPT를 사용하여 코드 분석"""
    system_message = SystemMessage(
        content="""
        당신은 전문적인 코드 리뷰어입니다. 주어진 코드를 분석하여 다음 항목들을 평가해주세요:
        
        1. 잘된 점:
        - 코드 구조
        - 네이밍 컨벤션
        - 모듈화/재사용성
        - 문서화
        
        2. 개선이 필요한 점:
        - 잠재적인 버그
        - 성능 이슈
        - 보안 취약점
        - 코드 스타일 개선사항
        
        각 항목에 대해 구체적인 예시와 개선 방안을 함께 제시해주세요.
        """
    )

    human_message = HumanMessage(content=str(contents))

    # messages 리스트로 직접 전달
    messages = [system_message, human_message]

    response = llm.invoke(messages)
    return response.content


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
                # 코드 가져오기
                contents = get_repo_contents(repo_url)

                # 코드 분석
                analysis = analyze_code(contents)

                # 결과 표시
                st.success("분석이 완료되었습니다!")
                st.markdown("## 코드 리뷰 결과")
                st.markdown(analysis)

            except Exception as e:
                st.error(f"분석 중 오류가 발생했습니다: {str(e)}")

# 사용 방법 안내
with st.expander("사용 방법"):
    st.markdown(
        """
    1. GitHub 레포지토리 URL을 입력창에 붙여넣으세요.
    2. 자동으로 코드 분석이 시작됩니다.
    3. 분석이 완료되면 코드의 장단점과 개선 방안을 확인할 수 있습니다.
    
    **참고**: 
    - 큰 레포지토리의 경우 분석에 시간이 걸릴 수 있습니다.
    - 현재는 Python, JavaScript, Java, C++, TypeScript 파일만 분석합니다.
    """
    )
