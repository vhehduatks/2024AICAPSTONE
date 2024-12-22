import tensorflow as tf
from body_parts_measurement_data_generator import BodyPartsMeasurementDataGenerator
from model.model import get_model
from loss import get_loss_fn
from tqdm import tqdm
import datetime
from tensorflow.keras import backend as K
import wandb

class BodyPartsMeasurementTrainer :
	def __init__(self, config):
		self.config = config
		self.data_generator_train = BodyPartsMeasurementDataGenerator(config, data_type="train")
		self.data_generator_validation = BodyPartsMeasurementDataGenerator(config, data_type="validation")
		self.model = get_model(config)
		self.loss_func_measurement = get_loss_fn(self.config["type_loss_fn"])
		self.loss_func_segmentation = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
		self.metric_func_measurement = tf.keras.metrics.mean_absolute_percentage_error
		if self.config['path_pretrained'] is not None :
			self.model.load_weights(self.config['path_pretrained'])
			print("pretrained model loaded from",self.config['path_pretrained'])
		else :
			print("train from scratch")
		# wandb.init(project='soulsoft_task')
		# wandb.config = config
			
	def start(self) :
		adam_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
		# self.model.compile(optimizer=adam_optimizer, loss=self.loss_func_measurement, metrics=["mape"])
		# callbacks = [
		#         tf.keras.callbacks.ModelCheckpoint(self.type_backbone + "-{epoch:02d}-{val_loss:.2f}.hdf5", save_best_only=True, mode="min")
		#     ]
		# self.model.fit(self.data_generator_train,epochs=100,steps_per_epoch=len(self.data_generator_train)/self.batch_size)
		# self.model.fit(self.data_generator_train,validation_data=self.data_generator_validation,epochs=100,steps_per_epoch=len(self.data_generator_train)/self.batch_size, callbacks=callbacks)

		current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		name_model = self.config['type_backbone']
		if self.config["type_attention"] != "none" :
			name_model += "_attention"
		if self.config["is_with_seg"] :
			name_model += "_seg"

		train_log_dir = 'logs/gradient_tape/' + name_model + "_" + current_time + '/loss'
		# validation_log_dir = 'logs/gradient_tape/' + name_model + "_" + current_time + '/test'
		train_summary_writer = tf.summary.create_file_writer(train_log_dir)
		# validation_summary_writer = tf.summary.create_file_writer(validation_log_dir)
		train_summary_writer.set_as_default()

		step = 0
		step_val = 0

		for epoch in range(int(self.config["epochs"])):
			tqdm_data_generator_train = tqdm(enumerate(self.data_generator_train), total=len(self.data_generator_train))
			tqdm_data_generator_validation = tqdm(enumerate(self.data_generator_validation), total=len(self.data_generator_validation))
			loss_epoch_train = 0
			mape_epoch_train = 0
			loss_epoch_val = 0
			mape_epoch_val = 0
			for batch_index, batch_data in tqdm_data_generator_train:
				with tf.GradientTape() as tape:
					if self.config["type_attention"] != "none" :
						preds = self.model([batch_data[0],batch_data[2]])
						losses = self.get_loss(preds, batch_data[1:])
					else :
						preds = self.model(batch_data[0])
						losses = self.get_loss(preds, batch_data[1:])
					if self.config["is_with_seg"]:
						loss = losses[0] + losses[1]
					else :
						loss = losses
					
					loss_epoch_train += loss
					grads = tape.gradient(loss, self.model.trainable_variables)
					adam_optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
					acc_mape = tf.reduce_mean(tf.keras.metrics.mean_absolute_percentage_error(batch_data[1], preds))
					mape_epoch_train += acc_mape

					if self.config["is_with_seg"]:
						tqdm_data_generator_train.set_description("epoch:{}/{},loss_measurement:{:.4f},loss_seg:{:.4f},acc_mape:{:.4f}".format(epoch, self.config["epochs"], losses[0], losses[1], acc_mape))
					else :
						tqdm_data_generator_train.set_description("epoch:{}/{},loss_measurement:{:.4f},acc_mape:{:.4f}".format(epoch, self.config["epochs"], loss, acc_mape))

					tf.summary.scalar('train_loss', loss, step=step)
					tf.summary.scalar('train_accuracy', acc_mape, step=step)
					# wandb.log({
					# 	'train_loss':loss,
					# 	'train_mape':acc_mape,
					# })
					step += 1
			

			print("loss_epoch_train",loss_epoch_train/len(self.data_generator_train))
			print("mape_epoch_train",mape_epoch_train/len(self.data_generator_train))
			if epoch % self.config["eval_term"] == 0 :
				for batch_index, batch_data in tqdm_data_generator_validation:
					if self.config["type_attention"] != "none" :
						batch_data
						preds = self.model([batch_data[0],batch_data[2]])
						losses = self.get_loss(preds, batch_data[1:])
					else :
						preds = self.model(batch_data[0])
						losses = self.get_loss(preds, batch_data[1:])
					if self.config["is_with_seg"]:
						loss = losses[0] + losses[1]
					else :
						loss = losses
					loss_epoch_val += loss
					acc_mape = tf.reduce_mean(tf.keras.metrics.mean_absolute_percentage_error(batch_data[1], preds))
					mape_epoch_val += acc_mape

					
					tf.summary.scalar('val_loss', loss, step=step_val)
					tf.summary.scalar('val_accuracy', acc_mape, step=step_val)
					# wandb.log({
					# 	'val_loss':loss,
					# 	'val_mape':acc_mape,
					# })
					step_val += 1
				print("loss_epoch_val",loss_epoch_val/len(self.data_generator_validation))
				print("mape_epoch_val",mape_epoch_val/len(self.data_generator_validation))

				name_full_model = name_model + "_" + str(epoch) + "_" + str(mape_epoch_val.numpy()/len(self.data_generator_validation))

				self.model.save_weights(name_full_model + ".h5")
				


	def get_loss(self, preds, batch_data) :
		if self.config["type_attention"] != "none" :
			if self.config["is_with_seg"] :
				batch_body_parts_measurement, batch_attention_features, batch_segs = batch_data
				pred_measurements, pred_segs = preds
			else :
				batch_body_parts_measurement, batch_attention_features = batch_data
				pred_measurements = preds
		else :
			if self.config["is_with_seg"] :
				batch_body_parts_measurement, batch_segs = batch_data
				pred_measurements, pred_segs = preds
			else :
				batch_body_parts_measurement = batch_data[0]
				pred_measurements = preds

		loss_measurements = self.loss_func_measurement(batch_body_parts_measurement, pred_measurements)
		if self.config["is_with_seg"] :
			loss_segs = self.loss_func_segmentation(batch_segs, pred_segs)
			return loss_measurements, loss_segs
		else :
			return loss_measurements

def mean_absolute_percentage_error(y_true, y_pred):

	diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
											K.epsilon(),
											None))

	return 100. * K.mean(diff, axis=-1)



if __name__ == "__main__":
	config = {
		# model
		'input_shape': [256,256,3],
		'batch_size': 4,
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
		'has_filename': False,

		# train
		'epochs': 30,
		'eval_term': 1
	}
	trainer = BodyPartsMeasurementTrainer(config)
	trainer.start()
