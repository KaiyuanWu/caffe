# coding: utf-8
import caffe
from caffe.proto import caffe_pb2
from find_face import FindFace
import  cv2
import mxnet as mx
import numpy as np
import importlib

from sklearn.metrics import pairwise_distances as pdg

class ImagePreprocessor():
    def __init__(self):
        self.img_height = 112
        self.img_width = 96

    def crop_image(self, img, landmarks = None, rect = None):
        coord5points = np.array([[30.2946, 65.5318, 48.0252, 33.5493, 62.7299], [
            51.6963, 51.5014, 71.7366, 92.3655, 92.2041]])
        T, _ = self.findNonreflectiveSimilarity(uv = landmarks, xy = coord5points)
        face = cv2.warpAffine(src=img, M=T, dsize=(self.img_width, self.img_height))
        face = (face - 127.5) / 128
        return np.rollaxis(face, 2, 0)

    # 从cp2tfrom.m 中 findSimilarity 翻译过来的
    def findNonreflectiveSimilarity(uv, xy):
        npnts = xy.shape[1]
        X = np.zeros((2 * npnts, 4))
        X[:npnts, :2] = xy[:2, :].T
        X[:npnts, 2] = 1
        X[npnts:, 0] = xy[1, :].T
        X[npnts:, 1] = -xy[0, :].T
        X[npnts:, 3] = 1
        U = uv.flatten()
        r, err, _, _ = np.linalg.lstsq(X, U)
        sc = r[0]
        ss = r[1]
        tx = r[2]
        ty = r[3]
        T = np.array([[sc, -ss, 0],
                      [ss, sc, 0],
                      [tx, ty, 1]])
        T = np.linalg.inv(T)
        return T[:, :2].T, err

class FeatureExtractor(object):
    def __init__(self,  cropper, executor, batch_size, imgdir_prefix, img_filenames, landmarks):
        assert type(cropper) == ImagePreprocessor
        self.cropper = cropper
        self.__set_executor(executor, batch_size)
        self.__set_imgs(imgdir_prefix, img_filenames, landmarks)
        assert len(executor.outputs) == 1
        assert  len(executor.outputs[0].shape) == 2
        self.num_feature = executor.outputs[0].shape[1]
        self.feature_buffer = np.zeros((self.num_imgs, self.num_feature))


    def __set_executor(self, executor, batch_size):
        input_shape = executor.arg_dict['data'].shape
        input_shape[0] = batch_size
        self.batch_size = batch_size
        self.executor = executor.reshape(allow_up_sizing=True,**input_shape)

    #预先crop好所有人脸图片
    def __set_imgs(self, imgdir_prefix, img_filenames, landmarks):
        img_height = self.cropper.img_height
        img_width = self.cropper.img_width
        img_channel = 3
        self.num_imgs = len(img_filenames)
        self.face_buffer = np.zeros((self.num_imgs, img_channel, img_height, img_width), dtype = 'float32')

        for idx  in range(self.num_imgs):
            imgfile = imgdir_prefix + img_filenames[idx]
            img = cv2.imread(imgfile, cv2.IMREAD_COLOR)
            self.face_buffer[idx]  = self.cropper.crop_image(img, landmarks = landmarks[idx])

    def extract_feature(self):
        input_data = self.executor.arg_dict['data']
        for ibatch_start in range(0, self.num_imgs, self.batch_size):
            pad = max(ibatch_start + self.batch_size - self.num_imgs, 0)
            input_data[ibatch_start:ibatch_start+self.batch_size - pad] = self.face_buffer[ibatch_start:ibatch_start+self.batch_size - pad]
            self.executor.forward(is_train=False)
            self.feature_buffer[ibatch_start:ibatch_start+self.batch_size - pad] =  self.executor.outputs[0].asnumpy()[:self.batch_size - pad]

        return self.feature_buffer

class ModelEvaluatorAllPair():
    def __init__(self, cropper, executor, batch_size, imgdir_prefix, img_filenames,  labels, landmarks, gal_index, probe_index):
        self.feature_extractor = FeatureExtractor(cropper, executor, batch_size, imgdir_prefix, img_filenames, landmarks)
        self.gal_index = gal_index
        self.probe_index = probe_index
        gal_label = labels[gal_index].reshape(-1,1)
        probe_label = labels[probe_index].reshape(1,-1)
        self.positive  = np.nonzero(gal_label == probe_label)
        self.negative = np.nonzero(gal_label != probe_label)


    def eval(self, farPoints = None):
        self.feature_extractor.extract_feature()
        feature = self.feature_extractor.feature_buffer
        gal_feat = feature[self.gal_index]
        probe_feat = feature[self.probe_index]
        score = pdg(gal_feat, probe_feat, metric='sqeuclidean')
        pos_score = score(self.positive[0], self.positive[1])
        neg_score = score(self.negative[0], self.negative[1])
        neg_score = np.sort(neg_score)

        num_negs = neg_score.shape[0]
        if farPoints is None:
            farPoints = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
        falseAlarms = int(farPoints*num_negs)

        thresholds = neg_score[falseAlarms]

        return np.mean(pos_score.reshape(-1,1), thresholds.reshape(1,-1), axis = 1)
