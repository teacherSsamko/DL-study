import base64
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


st.title("Image Chat Bot")
model = ChatOpenAI(model="gpt-4o-mini")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "base64_images" not in st.session_state:
    st.session_state.base64_images = None

# 이미지가 아직 업로드되지 않은 경우에만 파일 업로더 표시
if st.session_state.base64_images is None:
    if images := st.file_uploader(
        "저와 이야기 나눌 사진을 올려주세요!",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    ):
        for image in images:
            st.image(image)

        # 이미지를 base64로 변환하여 저장
        st.session_state.base64_images = [
            base64.b64encode(image.read()).decode("utf-8") for image in images
        ]
else:
    # 업로드된 이미지 표시
    for base64_image in st.session_state.base64_images:
        st.image(f"data:image/jpeg;base64,{base64_image}")

# 저장된 대화 내용 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 이미지가 업로드된 경우에만 채팅 입력 활성화
if st.session_state.base64_images:
    if prompt := st.chat_input("무슨 이야기가 하고 싶으신가요?"):
        # 사용자 메시지 저장 및 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            content = []
            for base64_image in st.session_state.base64_images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                )
            message = HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    *content,
                ],
            )
            result = model.invoke([message])
            response = result.content
            # 어시스턴트 응답 저장 및 표시
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)
