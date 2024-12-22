from tqdm import tqdm
import numpy as np
from model.model import get_model
from body_parts_measurement_data_generator import BodyPartsMeasurementDataGenerator
import datetime

class Evaluator :
	def __init__(self, data_generator, model):
		self.data_generator = data_generator
		self.model = model
		self.list_diff_parts_measurements = []
		self.list_mape = []
		self.list_mse_all = []
		self.list_mse_part = []
		self.list_diff_percentage_parts_measurements = []
		
	def run_evaluation(self) :
		for idx in tqdm(range(len(self.data_generator))) :
#         for idx in tqdm(range(10)) :
			batch_data = self.data_generator.__getitem__(idx)
			batch_images, batch_body_parts_measurement, batch_categorical_bmi_and_height = batch_data
			pd_parts_measurements = self.model.predict([batch_images, batch_categorical_bmi_and_height])
			batch_diff_parts_measurements = abs(batch_body_parts_measurement - pd_parts_measurements)
			batch_diff_percentage_parts_measurements = abs(batch_diff_parts_measurements / batch_body_parts_measurement)
			for each_diff_parts_measurements, each_diff_percentage_parts_measurements in zip(batch_diff_parts_measurements, batch_diff_percentage_parts_measurements) :
				self.list_diff_parts_measurements.append(each_diff_parts_measurements)
				self.list_diff_percentage_parts_measurements.append(each_diff_percentage_parts_measurements)

	def run_evaluation_new(self) :
		f_w = open("eval_results_test.txt","w")
		f_w.write("Evaluation Started At "+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")

		for idx in tqdm(range(len(self.data_generator))) :
			try :
				batch_data = self.data_generator.get_data(idx)
			except :
				print("idx",idx,"wrong")
			batch_images, batch_body_parts_measurement, batch_categorical_bmi_and_height = batch_data
			batch_images, batch_body_parts_measurement, batch_categorical_bmi_and_height = np.expand_dims(batch_images, axis=0), np.expand_dims(batch_body_parts_measurement, axis=0), np.expand_dims(batch_categorical_bmi_and_height, axis=0)

			# 31개 중 가슴 : 8, 허리 : 9, 엉덩이 :11, 허벅지:12
			pd_parts_measurements = self.model.predict([batch_images, batch_categorical_bmi_and_height])
			batch_diff_parts_measurements = abs(batch_body_parts_measurement - pd_parts_measurements)
			MSE_4 = (np.sum(np.power((batch_body_parts_measurement[:,[8,9,11,12]] - pd_parts_measurements[:,[8,9,11,12]]),2)))/(len(batch_body_parts_measurement)*4)
			MSE_ALL = (np.sum(np.power((batch_body_parts_measurement - pd_parts_measurements),2)))/(len(batch_body_parts_measurement)*31)
			batch_diff_percentage_parts_measurements = abs(batch_diff_parts_measurements / batch_body_parts_measurement)

			mape = sum(batch_diff_percentage_parts_measurements[0]) / len(batch_diff_percentage_parts_measurements[0])

			# str_filename = str(idx) + " filename: " + filename + "\n"
			str_gt = "gt: " + ", ".join(str(x) for x in batch_body_parts_measurement[0]) + "\n"
			str_pd = "pd: " + ", ".join(str(x) for x in pd_parts_measurements[0]) + "\n"
			str_pe = "ape: " + ", ".join(str(x) for x in batch_diff_percentage_parts_measurements[0]) + "\n"
			str_mape = "MAPE : " + str(mape) + "\n"
			str_mse_4 = "MSE_4 : " + str(MSE_4) + "\n"
			str_mse_all = "MSE_ALL : " + str(MSE_ALL) + "\n\n"

			# f_w.write(str_filename)
			f_w.write(str_gt)
			f_w.write(str_pd)
			f_w.write(str_pe)
			f_w.write(str_mape)
			f_w.write(str_mse_4)
			f_w.write(str_mse_all)

			self.list_mape.append(mape)
			self.list_mse_all.append(MSE_ALL)
			self.list_mse_part.append(MSE_4)
			self.list_diff_percentage_parts_measurements.append(batch_diff_percentage_parts_measurements)

		np_diff_percentage_parts_measurements = np.squeeze(np.array(self.list_diff_percentage_parts_measurements))
		for idx in range(np_diff_percentage_parts_measurements.shape[1]) :
			part_mape = sum(np_diff_percentage_parts_measurements[:,idx])/np_diff_percentage_parts_measurements.shape[0]
			f_w.write("Part MAPE "+ str(idx) + ": " + str(part_mape) + "\n")

		f_w.write("MAPE for Test Dataset : " + str(sum(self.list_mape)/len(self.list_mape)) + "\n")
		f_w.write("MSE_all for Test Dataset : " + str(sum(self.list_mse_all)/len(self.list_mse_all)) + "\n")
		f_w.write("MSE_part for Test Dataset : " + str(sum(self.list_mse_part)/len(self.list_mse_part)) + "\n")
		f_w.write("Evaluation Ended At "+datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
		f_w.close()
		
if __name__ == "__main__":
	config = {
		# model
		'input_shape': [256,256,3],
		'batch_size': 1,
		'path_pretrained': None, 
		'type_backbone': "blazepose", 
		
		# loss
		'type_loss_fn': 'wing',
		
		# data
		'seg_shape': [64,64],
		'path_classes': "./seg_classes.txt",
		'shuffle': True, 
		'is_normalized': False, 
		'is_with_seg': False, 
		'path_dataset': "D:\\data\\body_parts_measurement",
		## attention type
		'type_attention': "regression", #regression, categorical, none
		'num_category_bmi': 10,
		'num_category_height': 10,
		'has_filename': True,

		# train
		'epochs': 30,
		'eval_term': 1
	}
	model = get_model(config)
	model.summary()
	model.load_weights('blazepose_attention_0_3.2034787193590604.h5')
	data_generator_test = BodyPartsMeasurementDataGenerator(config, data_type="test")
	evaluator = Evaluator(data_generator_test, model)
	evaluator.run_evaluation_new()
	# np_diff_parts_measurements = np.squeeze(np.array(evaluator.list_diff_parts_measurements))
	# np_diff_percentage_parts_measurements = np.squeeze(np.array(evaluator.list_diff_percentage_parts_measurements))
	# list_part_mapes = []
	# for idx in range(np_diff_parts_measurements.shape[1]) :
	#     part_mape = sum(np_diff_percentage_parts_measurements[:,idx])/np_diff_parts_measurements.shape[0]
	#     print(idx, part_mape)
	#     #print(sum(np_diff_percentage_parts_measurements[:,idx])/np_diff_parts_measurements.shape[0])
	#     list_part_mapes.append(part_mape)
	# print("mape",sum(list_part_mapes)/len(list_part_mapes))

