import numpy as np
import tensorflow as tf
import glob
import cv2

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, input_shape=(192, 192, 3), batch_size=10, shuffle=True, data_type="Train", is_normalized = False):
#         csvfile = "/home/hangil/work/data/3d_pose_estimation/eccv18_posetrack_challenge/ECCV18_Challenge/Train/POSE/13032.csv"
        path_data_dir = "/home/hangil/work/data/3d_pose_estimation/eccv18_posetrack_challenge/ECCV18_Challenge"
        path_imgs = path_data_dir + "/"+ data_type +"/*/*.jpg"
        path_annos = path_data_dir + "/"+ data_type +"/*/*.csv"
        self.data_type = data_type
        self.list_path_imgs = glob.glob(path_imgs)
        self.list_path_annos = glob.glob(path_annos)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = True
        self.is_normalized = is_normalized
        
        # Refresh data-generator
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_path_imgs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_path_imgs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initailization
        X = np.empty((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        y = np.empty((self.batch_size, 8))
        
        # Generate data
        for idx_batch, idx_id in enumerate(indexes):
            X[idx_batch,], y[idx_batch,] = self.__get_data(idx_id)
        return X, y

    def __get_data(self, idx_id):
        path_img = self.list_path_imgs[idx_id]
        path_anno = self.list_path_annos[idx_id]
        img = cv2.imread(path_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
        pose3d = np.genfromtxt(path_anno, delimiter=',')
        measurements = self.get_measurements_from_kpts(pose3d)
        return img, measurements
        
    def get_measurements_from_kpts(self, pose3d) :
        dist_head = np.linalg.norm(pose3d[8]-pose3d[9])
        dist_shoulder = np.linalg.norm(pose3d[11]-pose3d[14])
        dist_body = np.linalg.norm(pose3d[0]-pose3d[7]) + np.linalg.norm(pose3d[7]-pose3d[8])
        dist_lleg = np.linalg.norm(pose3d[0]-pose3d[7]) + np.linalg.norm(pose3d[7]-pose3d[8])
        dist_rleg = np.linalg.norm(pose3d[0]-pose3d[7]) + np.linalg.norm(pose3d[7]-pose3d[8])
        dist_larm = np.linalg.norm(pose3d[0]-pose3d[7]) + np.linalg.norm(pose3d[7]-pose3d[8])
        dist_rarm = np.linalg.norm(pose3d[0]-pose3d[7]) + np.linalg.norm(pose3d[7]-pose3d[8])
        dist_height = dist_head + dist_body + (dist_lleg + dist_rleg)/2
        # normalize data
        if self.is_normalized :
            dist_head /= dist_height
            dist_shoulder /= dist_height
            dist_body /= dist_height
            dist_lleg /= dist_height
            dist_rleg /= dist_height
            dist_larm /= dist_height
            dist_rarm /= dist_height
            dist_height /= dist_height
        return np.array([dist_height, dist_head, dist_shoulder, dist_body, dist_lleg, dist_rleg, dist_larm, dist_rarm])
        
        
        