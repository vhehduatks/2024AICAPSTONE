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

        # 추론된 이미지 표시
        annotated_image_base64 = data.get("annotated_image")
        if annotated_image_base64:
            annotated_image_bytes = base64.b64decode(annotated_image_base64)
            annotated_image = Image.open(io.BytesIO(annotated_image_bytes))
            st.image(annotated_image, caption="YOLO Annotated Image", use_container_width=True)

        # 크롭된 이미지 섬네일 표시
        cropped_images = data.get("cropped_images", [])
        if cropped_images:
            st.subheader("Cropped Images")
            for image_path in cropped_images:
                with open(image_path, "rb") as f:
                    img_bytes = f.read()
                    img = Image.open(io.BytesIO(img_bytes))
                    st.image(img, width=100)

    else:
        st.error("이미지 처리 중 오류가 발생했습니다. 서버를 확인하세요.")
