import streamlit as st
import requests
from PIL import Image
import io
import base64

# FastAPI 서버 주소
FASTAPI_SERVER = "http://127.0.0.1:8000"

st.title("Camera Input")

# 카메라 입력
camera_input = st.camera_input("Capture an image")

# 추론 결과 섹션
if camera_input is not None:
    # 이미지를 서버로 전송
    with st.spinner("Processing image..."):
        response = requests.post(
            f"{FASTAPI_SERVER}/process_image",
            files={"file": camera_input.getvalue()},
        )

    if response.status_code == 200:
        data = response.json()

        # 크롭된 이미지 리스트 확인
        cropped_images_base64 = data.get("cropped_images", [])
        if cropped_images_base64:
            st.subheader("Cropped Images")
            # Base64로 인코딩된 이미지를 디코딩하여 표시
            cropped_images = [
                Image.open(io.BytesIO(base64.b64decode(image_base64)))
                for image_base64 in cropped_images_base64
            ]
            
            # 선택박스에 이미지 인덱스 제공
            options = [f"Image {i+1}" for i in range(len(cropped_images))]
            selected_option = st.selectbox("Select an image to display:", options)
            
            # 선택된 이미지 표시
            selected_index = options.index(selected_option)
            selected_image = cropped_images[selected_index]
            st.image(selected_image, caption=f"Selected Image: {selected_option}", use_container_width=True)
            
            # 키와 몸무게 입력 단계
            st.subheader("Enter Your Information")
            height = st.text_input("Enter your height (cm):", "")
            weight = st.text_input("Enter your weight (kg):", "")
            
            # 선택된 이미지를 백엔드로 전송
            if st.button("Send Selected Image with Info to Backend"):
                if not height or not weight:
                    st.warning("Please enter both height and weight.")
                else:
                    with st.spinner("Sending image and information to backend..."):
                        # 이미지를 BytesIO로 변환
                        buffer = io.BytesIO()
                        selected_image.save(buffer, format="JPEG")
                        buffer.seek(0)
                        
                        # 백엔드로 전송
                        response = requests.post(
                            f"{FASTAPI_SERVER}/predict_image",
                            files={"file": buffer},
                            data={"height": height, "weight": weight},
                        )
                    
                    if response.status_code == 200:
                        st.success("Image and information successfully sent to the backend!")
                        st.json(response.json())  # 백엔드 응답 표시
                    else:
                        st.error("Failed to send image and information to the backend.")
        else:
            st.warning("No cropped images received from the server.")
    else:
        st.error("이미지 처리 중 오류가 발생했습니다. 서버를 확인하세요.")