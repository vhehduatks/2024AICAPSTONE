import numpy as np
import pandas as pd
import os
import glob
import cv2
import tensorflow as tf
import json
import codecs

class BodyPartsMeasurementDataGenerator(tf.keras.utils.Sequence) :
    def __init__(self, config=None, data_type="train"):
        self.batch_size = config['batch_size']
        self.input_shape = config['input_shape']
        self.seg_shape = config['seg_shape']
        self.path_classes = config['path_classes']
        self.type_backbone = config['type_backbone']
        self.type_loss_fn = config['type_loss_fn']
        self.epochs = config['epochs']
        self.is_with_seg = config['is_with_seg']
        self.path_dataset = config['path_dataset']
        self.data_type = data_type
        self.shuffle = config['shuffle']
        self.is_normalized = config['is_normalized']
        self.type_attention = config['type_attention']
        self.num_category_bmi = config['num_category_bmi']
        self.num_category_height = config['num_category_height']
        self.has_filename = config['has_filename']

        with codecs.open(self.path_classes, encoding='utf-8') as f_r :
            classes = f_r.readlines()
        self.classes = [each_class.rstrip('\r\n') for each_class in classes]
        self.num_seg_channels = len(self.classes)
        self.get_list_data()
        self.on_epoch_end()
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        return self.data_generation(indexes)
    
    def data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initailization
        batch_images = np.empty((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]), dtype=np.float32)
        batch_body_parts_measurement = np.empty((self.batch_size, 31), dtype=np.float32)

        if self.type_attention == "categorical" :
            batch_attention_features = np.empty((self.batch_size, self.num_category_bmi + self.num_category_height), dtype=np.float32)
        elif self.type_attention == "regression" :
            batch_attention_features = np.empty((self.batch_size, 2), dtype=np.float32)

        if self.is_with_seg :
            batch_segs = np.empty((self.batch_size, self.input_shape[0]//2, self.input_shape[1]//2, 1), dtype=np.float32)
        
        # Generate data
        for idx_batch, idx_id in enumerate(indexes):
            try :
                if self.type_attention != "none" :
                    if self.is_with_seg :
                        batch_images[idx_batch,], batch_body_parts_measurement[idx_batch,], batch_attention_features[idx_batch,], batch_segs[idx_batch,] = self.get_data(idx_id)
                    else :
                        batch_images[idx_batch,], batch_body_parts_measurement[idx_batch,], batch_attention_features[idx_batch,] = self.get_data(idx_id)
                else :
                    if self.is_with_seg :
                        batch_images[idx_batch,], batch_body_parts_measurement[idx_batch,], batch_segs[idx_batch,] = self.get_data(idx_id)
                    else :
                        batch_images[idx_batch,], batch_body_parts_measurement[idx_batch,] = self.get_data(idx_id)
            except Exception as e: 
                print("error",str(e))

        list_outputs = []
        list_outputs.append(batch_images)
        list_outputs.append(batch_body_parts_measurement)
        if self.type_attention != "none" :
            list_outputs.append(batch_attention_features)
        if self.is_with_seg :
            list_outputs.append(batch_segs)
        return list_outputs

    def get_data(self, idx_id):
        list_outputs = []
        path_img = self.list_data[idx_id][0]
        path_json = self.list_data[idx_id][1]
        body_parts_measurement = self.list_data[idx_id][2]
        img = cv2.imread(path_img)
        if self.is_with_seg :
            img_shape = img.shape
            segs = self.get_segmentation_map(path_json, img_shape, self.input_shape)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))

        if self.is_normalized :
            img = img / 255.0
            body_parts_measurement = (body_parts_measurement - self.min_body_parts_measurement) / (self.max_body_parts_measurement - self.min_body_parts_measurement)

        if self.type_attention == "regression" :
            attention_features = self.get_regression_features(body_parts_measurement)
        
        body_parts_measurement = body_parts_measurement[1:32]            
        body_parts_measurement = np.array(body_parts_measurement)

        list_outputs.append(img)
        list_outputs.append(body_parts_measurement)
        if self.type_attention != "none" :
            list_outputs.append(attention_features)
        if self.is_with_seg :
            list_outputs.append(segs)

        if self.has_filename :
            list_outputs.append(path_img)

        return list_outputs

    def get_regression_features(self, body_parts_measurement) :
        height = body_parts_measurement[0]
        bmi = body_parts_measurement[36]
        regression_features = np.asarray([height, bmi])
        return regression_features

    
    def get_categorical_bmi_and_height(self, body_parts_measurement) :
        bmi = body_parts_measurement[36]
        categorical_bmi = np.zeros(self.num_category_bmi)
        category_label_bmi = int((bmi - self.min_body_parts_measurement[36]) / self.unit_bmi) - 1
        categorical_bmi[category_label_bmi] = 1

        height = body_parts_measurement[0]
        categorical_height = np.zeros(self.num_category_height)
        category_label_height = int((height - self.min_body_parts_measurement[0]) / self.unit_height) - 1
        categorical_height[category_label_height] = 1

        categorical_bmi_and_height = np.concatenate([categorical_bmi, categorical_height])

        return categorical_bmi_and_height
    
    def get_categorical_height(self, height) :
        categorical_height = np.zeros(self.num_category_height)
        category_label = int((height - self.min_body_parts_measurement[0]) / self.unit_height) - 1
        categorical_height[category_label] = 1
        return categorical_height

    def get_segmentation_map(self, filename, img_shape, input_shape) :
        codec_f = codecs.open(filename, 'r', encoding='utf-8')
        # codec_f = codecs.open(filename, 'r', encoding='ISO-8859-1')
        
        data = json.load(codec_f)
        codec_f.close()

        segs = np.zeros((img_shape[0],img_shape[1],1))
        for each_labeling_info in data['labelingInfo'] :
            polygon = each_labeling_info['polygon']
            coordinates = polygon['location'].split(" ")[:-1]
            list_coords = [int(x) for x in coordinates]
            contours = np.array(list_coords).reshape(-1,2)
            label = polygon['label']
            index_label = self.classes.index(label)
            cv2.fillPoly(segs, pts=[contours], color=(index_label,index_label,index_label))
        segs = cv2.resize(segs, (input_shape[0]//2,input_shape[1]//2))
        segs = np.expand_dims(segs, axis=2)
        return segs

    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_data) / self.batch_size))
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def get_list_data(self) :
        list_user_dirs = os.listdir(self.path_dataset + os.path.sep + self.data_type)
        list_user_dirs.sort()
        print("list_user_dirs",list_user_dirs)
        self.list_data = []
        self.min_body_parts_measurement = np.full(37, 10000, dtype=np.float32)
        self.max_body_parts_measurement = np.zeros(37, dtype=np.float32)
        for user_dir in list_user_dirs :
            try :
                path_full_user_dir = self.path_dataset + os.path.sep + self.data_type + os.path.sep + user_dir
                ##### get parts measurement
                user_dir_token = user_dir.split('_')
                if user_dir_token[0] == 'TL':
                    continue
                path_user_csv = path_full_user_dir + os.path.sep + "csv" + os.path.sep + user_dir + ".csv"
                df = pd.read_csv(path_user_csv, encoding = "cp949")
                body_parts_measurement = list(df.iloc[[1]].values[0][3:-1])
                # change gender to digit
                if body_parts_measurement[-2] == "F" : 
                    body_parts_measurement[-2] = "1"
                elif body_parts_measurement[-2] == "M" :
                    body_parts_measurement[-2] = "0"
                else :
                    print("wrong in ",user_dir)
                # get bmi
                height = float(body_parts_measurement[0])
                weight = float(body_parts_measurement[32])
                bmi = weight / (height/100. * height/100.)
                body_parts_measurement.append(bmi)
                body_parts_measurement = np.array(body_parts_measurement, dtype=np.float32)
                # body_parts_measurement = body_parts_measurement.astype(np.float32)
                # get min and max value for each column for normalization
                for idx, (min_val, max_val, measurement) in enumerate(zip(self.min_body_parts_measurement, self.max_body_parts_measurement, body_parts_measurement)) :
                    if measurement > max_val :
                        self.max_body_parts_measurement[idx] = measurement
                    if measurement < min_val :
                        self.min_body_parts_measurement[idx] = measurement

                ##### get list of images
                path_user_image_dir = path_full_user_dir + os.path.sep + "image"
                # get only images for attention posture
                list_path_images = glob.glob(path_user_image_dir + os.path.sep + "*.jpg")
                # list_path_images = glob.glob(path_user_image_dir + os.path.sep + "04_03_*_01.jpg")
                for each_path_image in list_path_images :
                    each_path_json = each_path_image.replace("jpg","json")
                    each_path_json = each_path_json.replace("image","json")
                    each_path_json = each_path_json.split(os.path.sep)
                    each_path_json[-3] = 'TL_'+each_path_json[-3]
                    each_path_json = os.path.sep.join(each_path_json)
                    
                    each_data = [each_path_image, each_path_json, body_parts_measurement]
                    self.list_data.append(each_data)
            except Exception as e: 
                print("error in user_dir",user_dir,str(e))
                print("body_parts_measurement",body_parts_measurement)
        
        # default settings
        self.min_body_parts_measurement[-3] = 0
        self.avg_bmi = (self.min_body_parts_measurement[36] + self.max_body_parts_measurement[36]) / 2
        self.avg_height = (self.min_body_parts_measurement[0] + self.max_body_parts_measurement[0]) / 2
        self.unit_bmi = (self.max_body_parts_measurement[36] - self.min_body_parts_measurement[36]) / self.num_category_bmi
        self.unit_height = (self.max_body_parts_measurement[0] - self.min_body_parts_measurement[0]) / self.num_category_height

        # self.min_body_parts_measurement = self.min_body_parts_measurement[1:32]  
        # self.max_body_parts_measurement = self.max_body_parts_measurement[1:32]  

        # print("self.min_body_parts_measurement",self.min_body_parts_measurement)
        # print("self.max_body_parts_measurement",self.max_body_parts_measurement)
        # print("self.avg_bmi",self.avg_bmi)
        # print("self.avg_bmi",self.avg_bmi)
        # print("self.avg_height",self.avg_height)
        # print("self.unit_bmi",self.unit_bmi)
        # print("self.unit_height",self.unit_height)


