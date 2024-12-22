import tensorflow as tf
from model.blazepose import BlazePose
from model.mobilenet_v3 import MobileNetV3
from model.measurement_attention_mlp import get_measurement_attention_mlp
# from model.blazepose_full import BlazePose

def get_model(config) :

	attention_mlp = get_measurement_attention_mlp(num_input_features=2)

	blazepose_model = BlazePose(input_shape=[256,256,3], num_keypoints=31, num_seg_channels=10, attention_model=attention_mlp)
	model = blazepose_model.build_model(model_type="REGRESSION")

	return model

