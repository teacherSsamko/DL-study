import os

import streamlit as st
from github import Github
from github.GithubException import UnknownObjectException
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
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
                elif content.path.endswith((".py", ".js", ".java", ".cpp", ".ts")):
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

                # 코드 분석
                analysis = analyze_code(contents, readme_content)

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
