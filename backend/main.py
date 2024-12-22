from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import os
import base64


import sys
sys.path.append("..")
from body_part_measurement.model.measurement_attention_mlp import get_measurement_attention_mlp
from body_part_measurement.model.blazepose import BlazePose


app = FastAPI()

# YOLO 모델 로드
yolo_model = YOLO('yolo11n-pose.pt')


# blazepose model
# model
attention_mlp = get_measurement_attention_mlp(batch_size=1,num_input_features=2)
blazepose_model = BlazePose(batch_size=1,input_shape=[256,256,3], num_keypoints=31, num_seg_channels=10, attention_model=attention_mlp)
body_model = blazepose_model.build_model(model_type="REGRESSION")
body_model.load_weights(r'C:\AI_class_yang\AI_code\신체 치수 예측\body_part_measurement_source\body_part_measurement\scripts\blazepose_attention_0_3.022645455819589.h5')



# 이미지 저장 경로
SAVE_DIR = r"C:\AI_class_yang\AI_code\신체 치수 예측\body_part_measurement_source\backend\captures"
os.makedirs(SAVE_DIR, exist_ok=True)

@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
	try:
		# 이미지를 읽어서 OpenCV 형식으로 변환
		contents = await file.read()
		np_image = np.frombuffer(contents, np.uint8)
		frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

		# 모델 추론
		results = yolo_model.predict(frame, conf=0.5)

		# 결과 이미지
		annotated_frame = results[0].plot()

		# 크롭된 이미지 처리
		cropped_images_name = []
		cropped_images = []
		for result in results:
			if hasattr(result, 'keypoints') and result.keypoints is not None:
				keypoints_data = result.keypoints.data[0]  # 첫 번째 객체의 키포인트 데이터

				try:
					head_conf = keypoints_data[0, 2]
					left_hand_conf = keypoints_data[9, 2]
					right_hand_conf = keypoints_data[10, 2]
					left_leg_conf = keypoints_data[15, 2]  
					right_leg_conf = keypoints_data[16, 2]  

					if (
						head_conf > 0.5 and
						left_hand_conf > 0.5 and
						right_hand_conf > 0.5 and
						left_leg_conf > 0.5 and
						right_leg_conf > 0.5
					):	
						print("모든 신체 부위가 감지되었습니다.")
	
						for box in result.boxes.data.tolist():
							x1, y1, x2, y2, confidence, class_id = map(int, box[:6])
							if class_id == 0:  # 사람 클래스
								# 이미지 크롭
								cropped_img = frame[y1:y2, x1:x2]
								_, encoded_croped_image = cv2.imencode(".jpg", cropped_img)
								encoded_croped_image_base64 = base64.b64encode(encoded_croped_image).decode("utf-8")
								cropped_images.append(encoded_croped_image_base64)
								cropped_file_name = os.path.join(SAVE_DIR, f"cropped_person_{x1}_{y1}.jpg")
								cv2.imwrite(cropped_file_name, cropped_img)
								cropped_images_name.append(cropped_file_name)
				except:
					pass

		# 결과 이미지 Base64 인코딩
		_, encoded_image = cv2.imencode(".jpg", annotated_frame)
		annotated_image_base64 = base64.b64encode(encoded_image).decode("utf-8")

		return {
			"annotated_image_all": annotated_image_base64,
			"cropped_images" : cropped_images,
			"cropped_images_name": cropped_images_name,
		}

	except Exception as e:
		return JSONResponse({"error": str(e)}, status_code=500)


# response = requests.post(
#     f"{FASTAPI_SERVER}/predict_image",
#     files={"file": buffer},
#     data={"height": height, "weight": weight},
# )

@app.post("/predict_image")
async def predict_image(
	file: UploadFile = File(...),    
	height: float = Form(...),
	weight: float = Form(...),
	):
	try:
		contents = await file.read()
		np_image = np.frombuffer(contents, np.uint8)
		frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

		if frame is None:
			raise ValueError("Invalid image data")
		
		bmi = weight / (height / 100) ** 2

		img = cv2.resize(frame, (256, 256))
		test_image = img[np.newaxis, :]  # shape to (1, 256, 256, 3)

		test_bmi_and_height = np.asarray([height, bmi])[np.newaxis, :]

		total_measurement = body_model.predict([test_image, test_bmi_and_height])

		return {
			"total_measurement": total_measurement.tolist(),
			"chest_measurement" : total_measurement[0][8].tolist(),
			"waist_measurement" : total_measurement[0][9].tolist(),
			"hip_measurement" : total_measurement[0][11].tolist(),
			"thigh_measurement" : total_measurement[0][12].tolist(),
		}

	except Exception as e:
		return JSONResponse({"error": str(e)}, status_code=500)