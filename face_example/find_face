# -*- coding: utf-8 -*-
#利用MTCNN算法寻找人脸
__author__ = "kaiwu"
__date__ = "$Oct 15, 2016 4:40:51 PM$"
import caffe
from caffe.proto import caffe_pb2
import numpy as np
import cv2

class FindFace:
	def __init__(self, PNet,RNet,ONet,LNet):
		self.PNet = caffe.Net(PNet['deploy_proto'], PNet['model'],caffe.TEST)
		self.RNet = caffe.Net(RNet['deploy_proto'], RNet['model'],caffe.TEST)
		self.ONet = caffe.Net(ONet['deploy_proto'], ONet['model'],caffe.TEST)
		self.LNet = caffe.Net(LNet['deploy_proto'], LNet['model'],caffe.TEST)
		self.threshold = [0.6, 0.7, 0.7]
		self.fastresize = False
		self.factor = 0.709
		self.roll_params()
	
	def detect_face(self, img, minisize):
		factor_count = 0
		total_boxes = []
		points = []
		h = img.shape[0]
		w = img.shape[1]
		c = img.shape[2]
		assert c==3
		minl = min(w, h)
		img= img.astype('float32')
		if self.fastresize:
			im_data=(img-127.5)*0.0078125
		m = 12./minisize
		minl = minl*m
		#create scale pyramid
		scales=[]
		while minl >= 12:
			scales += [m*(self.factor**factor_count)]
			minl = minl*self.factor
			factor_count = factor_count+1
	
		#first stage
		for j in range(len(scales)):
			scale = scales[j]
			hs = int(np.ceil(h*scale))
			ws = int(np.ceil(w*scale))
			if self.fastresize:
				im_data = cv2.resize(im_data,(ws,hs), interpolation= cv2.INTER_LINEAR)
			else:
				im_data= (cv2.resize(img,(ws, hs), interpolation= cv2.INTER_LINEAR)-127.5)*0.0078125

			self.PNet.blobs['data'].reshape(1,3,hs,ws)
			self.PNet.blobs['data'].data[0,...] = np.rollaxis(im_data,2,0)[[2,1,0]]
			out = self.PNet.forward()
			boxes = self.generateBoundingBox(out['prob1'][0,1,...], out['conv4-2'][0,...], scale, self.threshold[0])
			#inter-scale nms
			pick=self.nms(boxes, 0.5, 'Union')
			if len(pick) > 0:
				total_boxes.append(boxes[pick])

		total_boxes = np.concatenate(total_boxes)
		numbox = total_boxes.shape[0]
		if numbox > 0:
			pick = self.nms(total_boxes, 0.7, 'Union')
			total_boxes = total_boxes[pick]
			bbw = total_boxes[:,2] - total_boxes[:,0]
			bbh = total_boxes[:,3] - total_boxes[:,1]
			total_boxes[:,0] += total_boxes[:,5]*bbw
			total_boxes[:,1] += total_boxes[:,6]*bbh
			total_boxes[:,2] += total_boxes[:,7]*bbw
			total_boxes[:,3] += total_boxes[:,8]*bbh
			total_boxes = self.rerec(total_boxes)
			total_boxes = total_boxes[:,:4].astype('int32')
			dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = self.pad(total_boxes,w,h)

			#second stage
			tempimg = np.zeros((numbox, 3, 24, 24))
			for k in range(numbox):
				tmp = np.zeros((tmph[k],tmpw[k],3))
				tmp[dy[k]:edy[k],dx[k]:edx[k],:]  = img[y[k]:ey[k], x[k]:ex[k],:]
				tempimg[k, :,:,:] = np.rollaxis(cv2.resize(tmp,(24, 24)), 2, 0)[[2,1,0]]
			self.RNet.blobs['data'].reshape(numbox, 3, 24, 24);
			self.RNet.blobs['data'].data[:] = (tempimg-127.5)*0.0078125
			out = self.RNet.forward()
			score = out['prob1'][...,1]
			index = score > self.threshold[1]
			total_boxes =np.concatenate((total_boxes[index,:4],score[index].reshape(-1,1)), axis=1)
			mv = out['conv5-2'][index]
			numbox = total_boxes.shape[0]
			if numbox > 0:
				pick = self.nms(total_boxes,0.7,'Union')
				total_boxes = total_boxes[pick,:]
				total_boxes = self.bbreg(total_boxes,mv[pick,:])
				total_boxes = self.rerec(total_boxes)
				total_boxes = total_boxes.astype('int32')

				dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = self.pad(total_boxes, w, h)
				#third stage
				numbox = total_boxes.shape[0]
				tempimg = np.zeros((numbox, 3, 48,48))
				for k in range(numbox):
					tmp = np.zeros((tmph[k],tmpw[k],3))
					tmp[dy[k]:edy[k],dx[k]:edx[k],:] =img[y[k]:ey[k],x[k]:ex[k],:]
					tempimg[k,...] = np.rollaxis(cv2.resize(tmp,(48,48)),2, 0)[[2,1,0]]

				self.ONet.blobs['data'].reshape(numbox, 3, 48, 48)
				self.ONet.blobs['data'].data[:] = (tempimg-127.5)*0.0078125
				out = self.ONet.forward()
				score = out['prob1'][:,1]
				points=out['conv6-3']
				index = score > self.threshold[2]
				points = points[index,:]
				total_boxes= np.concatenate((total_boxes[index,:4], score[index].reshape(-1,1)), axis=1)
				mv =  out['conv6-2'][index]
				bbw = (total_boxes[:,2]-total_boxes[:,0]+1).reshape(-1,1)
				bbh = (total_boxes[:,3]-total_boxes[:,1]+1).reshape(-1,1)
				points[:, :5] = bbw*points[:,:5] + total_boxes[:,0].reshape(-1,1) -1
				points[:, 5:] = bbh*points[:, 5:] + total_boxes[:,1].reshape(-1,1) -1
				if total_boxes.shape[0] >0:
					total_boxes = self.bbreg(total_boxes,mv)
					pick = self.nms(total_boxes, 0.7, 'Min')
					total_boxes=total_boxes[pick,:]
					points=points[pick,:]
					numbox=total_boxes.shape[0]
					#extended stage
					if numbox > 0:
						tempimg= np.zeros((numbox, 24, 24, 15))
						patchw = self.max2(total_boxes[:,2] - total_boxes[:,0] + 1, total_boxes[:,3] - total_boxes[:,1] + 1)
						patchw = (patchw*0.25).astype('int32')
						tmp = (patchw%2)==1
						patchw[tmp] = patchw[tmp]+1
						pointx = np.ones((numbox,5))
						pointy = np.ones((numbox,5))
						for k in  range(5):
							x=(points[:,k] - 0.5*patchw).astype('int32')
							y=(points[:,k+5] - 0.5*patchw).astype('int32')
							dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph= self.pad( np.vstack((x, y, x+patchw, y+patchw)).T,w,h)
							for j in range(numbox):
								tmpim = np.zeros((tmpw[j],tmpw[j],3))
								tmpim[dy[j]:edy[j],dx[j]:edx[j],:] = img[y[j]:ey[j],x[j]:ex[j],:]
								tempimg[j, :,:,k*3:k*3+3]= cv2.resize(tmpim,(24, 24))[...,[2,1,0]]

						self.LNet.blobs['data'].reshape(numbox, 15, 24, 24)
						self.LNet.blobs['data'].data[:] = np.rollaxis((tempimg-127.5)*0.0078125, 3, 1)
						out=self.LNet.forward()

						for k in range(5):
							layername = "fc5_%d"%(k+1)
							#do not make a large movement
							temp = abs(out[layername][:,0]-0.5) > 0.35
							out[layername][temp,:] = 0.5

							temp =  abs(out[layername][:,1]-0.5) > 0.35
							out[layername][temp, :] = 0.5

							pointx[:,k]=points[:,k] - 0.5*patchw + out[layername][:,0]*patchw
							pointy[:,k]=points[:,k+5] - 0.5*patchw + out[layername][:,1]*patchw

						for j in range(numbox):
							points[j,:5] = pointx[j,:]
							points[j,5:] = pointy[j,:]
		return total_boxes, points

	def roll_params(self):
		#旋转卷积参数的横竖方向
		pnet_names = ['conv1', 'conv2','conv3', 'conv4-1', 'conv4-2']
		for name in pnet_names:
			self.PNet.params[name][0].data[:] = np.rollaxis(self.PNet.params[name][0].data, 3, 2)
		rnet_names = ['conv1', 'conv2', 'conv3']
		for name in rnet_names:
			self.RNet.params[name][0].data[:] = np.rollaxis(self.RNet.params[name][0].data, 3, 2)
		#特殊处理一下从卷积到全连接部分的权重
		name = 'conv4'
		input_shape = (64,3,3)
		num_outputs = 128
		temp_param = self.RNet.params[name][0].data.copy()
		temp_param = temp_param.reshape(num_outputs, input_shape[0],input_shape[1],input_shape[2])
		temp_param = np.rollaxis(temp_param, 3,2)
		temp_param = temp_param.reshape(num_outputs, input_shape[0]*input_shape[1]*input_shape[2])
		self.RNet.params[name][0].data[:] = temp_param

		onet_names = ['conv1', 'conv2', 'conv3', 'conv4']
		for name in onet_names:
			self.ONet.params[name][0].data[:] = np.rollaxis(self.ONet.params[name][0].data, 3, 2)

		name = "conv5"
		input_shape = (128, 3, 3)
		num_outputs = 256
		temp_param = self.ONet.params[name][0].data.copy()
		temp_param = temp_param.reshape(num_outputs, input_shape[0], input_shape[1], input_shape[2])
		temp_param = np.rollaxis(temp_param, 3, 2)
		temp_param = temp_param.reshape(num_outputs, input_shape[0] * input_shape[1] * input_shape[2])
		self.ONet.params[name][0].data[:] = temp_param

		lnet_names = ['conv1_1', 'conv2_1', 'conv3_1', 'conv1_2', 'conv2_2',
					  'conv3_2', 'conv1_3', 'conv2_3', 'conv3_3', 'conv1_4',
					  'conv2_4', 'conv3_4', 'conv1_5', 'conv2_5', 'conv3_5']
		for name in lnet_names:
			self.LNet.params[name][0].data[:] = np.rollaxis(self.LNet.params[name][0].data, 3, 2)
		name = "fc4"
		input_shape = (320, 3, 3)
		num_outputs = 256
		temp_param = self.LNet.params[name][0].data.copy()
		temp_param = temp_param.reshape(num_outputs, input_shape[0], input_shape[1], input_shape[2])
		temp_param = np.rollaxis(temp_param, 3, 2)
		temp_param = temp_param.reshape(num_outputs, input_shape[0] * input_shape[1] * input_shape[2])
		self.LNet.params[name][0].data[:] = temp_param



	def pad(self, total_boxes, w, h):
		#compute the padding coordinates (pad the bounding boxes to square)
		tmpw=total_boxes[:,2]-total_boxes[:,0]+1
		tmph=total_boxes[:,3]-total_boxes[:,1]+1
		numbox=total_boxes.shape[0]

		dx = np.zeros(numbox,dtype = 'int32')
		dy = np.zeros(numbox,dtype = 'int32')
		edx = tmpw
		edy = tmph

		x = total_boxes[:,0]
		y = total_boxes[:,1]
		ex= total_boxes[:,2]+1
		ey= total_boxes[:,3]+1

		tmp = ex > w
		edx[tmp] = -ex[tmp] + w + tmpw[tmp]
		ex[tmp] = w

		tmp = ey>h
		edy[tmp] = -ey[tmp] + h + tmph[tmp]
		ey[tmp] = h

		tmp = x<0
		dx[tmp] =  -x[tmp]
		x[tmp] = 0

		tmp = y<0
		dy[tmp] =  -y[tmp]
		y[tmp] = 0

		return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph
		    
	def generateBoundingBox(self, map, reg, scale, threshold):
		#use heatmap to generate bounding boxes
		stride = 2
		cellsize = 12

		index = (map >= threshold)
		y,x = np.nonzero(index)
		score = map[index]
		reg = reg[:,index]

		boundingbox = np.array([x,y]).T
		boundingbox = np.concatenate([((stride*boundingbox+1)/scale).astype('int32')-1, ((stride*boundingbox+cellsize-1+1)/scale).astype('int32')-1, score.reshape(-1,1), reg.T], axis = 1)
		return boundingbox
    
	def nms(self, boxes,threshold,type):
		#NMS
		if boxes.shape[0] == 0:
			return np.array([])
		x1 = boxes[:,0]
		y1 = boxes[:,1]
		x2 = boxes[:,2]
		y2 = boxes[:,3]
		s = boxes[:,4]
		area = (x2-x1+1) * (y2-y1+1)
		I = np.argsort(s)
		pick = []
		while I.shape[0] > 0:
			i = I[-1]
			pick.append(i)
			xx1 = self.max1(x1[i], x1[I[:-1]])
			yy1 = self.max1(y1[i], y1[I[:-1]])
			xx2 = self.min1(x2[i], x2[I[:-1]])
			yy2 = self.min1(y2[i], y2[I[:-1]])
			w = self.max1(0.0, xx2-xx1+1)
			h = self.max1(0.0, yy2-yy1+1)
			inter = w*h
			if type == 'Min':
				o = inter / self.min1(area[i],area[I[:-1]])
			else:
				o = inter / (area[i] + area[I[:-1]] - inter)
			I = I[np.nonzero(o<=threshold)]
		return pick
    
	def rerec(self, bboxA):
		#convert bboxA to square
		h = bboxA[:,3] - bboxA[:,1]
		w = bboxA[:,2] - bboxA[:,0]
		l = self.max2(w, h)
		bboxA[:,0] += w*0.5-l*0.5
		bboxA[:,1] += h*0.5-l*0.5
		bboxA[:,2:4] = bboxA[:,0:2] + l.reshape(-1,1)
		return bboxA

	def max1(self, x, v):
		o = v.copy()
		o[x > v] = x
		return o

	def min1(self, x, v):
		o = v.copy()
		o[x < v] = x
		return o

	def max2(self, x, y):
		assert  x.shape[0] == y.shape[0]
		o = x.copy()
		index = x < y
		o[ index ] = y[index]
		return o

	def min2(self, x, y):
		assert x.shape[0] == y.shape[0]
		o = x.copy()
		index = x>y
		o[index] = y[index]
		return o

	def bbreg(self,boundingbox,reg):
		#calibrate bouding boxes
		if reg.shape[1] == 1:
			reg = reg.reshape(reg.shape[2], reg.shape[3])
		w = boundingbox[:,2] - boundingbox[:,0] + 1
		h = boundingbox[:,3] - boundingbox[:,1] + 1
		boundingbox[:, 0] += reg[:, 0]*w
		boundingbox[:, 1] += reg[:, 1]*h
		boundingbox[:, 2] += reg[:, 2] * w
		boundingbox[:, 3] += reg[:, 3] * h
		return boundingbox



model_prefix = 'MTCNNv2/model/'
PNet = {"deploy_proto": model_prefix + "det1.prototxt",
		"model": model_prefix + "det1.caffemodel"}
RNet = {"deploy_proto": model_prefix + "det2.prototxt",
		"model": model_prefix + "det2.caffemodel"}
ONet = {"deploy_proto": model_prefix + "det3.prototxt",
		"model": model_prefix + "det3.caffemodel"}
LNet = {"deploy_proto": model_prefix + "det4.prototxt",
		"model": model_prefix + "det4.caffemodel"}

"""
PNet = caffe.Net(PNet['deploy_proto'], PNet['model'], caffe.TEST)
RNet = caffe.Net(RNet['deploy_proto'], RNet['model'], caffe.TEST)
ONet = caffe.Net(ONet['deploy_proto'], ONet['model'], caffe.TEST)
LNet = caffe.Net(LNet['deploy_proto'], LNet['model'], caffe.TEST)
"""
caffe.set_mode_cpu()
find_face = FindFace(PNet,RNet,ONet,LNet)
imglist = ['test4.jpg', 'test3.bmp', 'test1.jpg', 'test2.jpg']
img_dir_prefix = "MTCNNv2/"

for imgname in imglist:
	img = cv2.imread(img_dir_prefix + imgname)
	minl = min(img.shape[0], img.shape[1])
	minsize = int(minl * 0.1)
	rects, pnts = find_face.detect_face(img, minsize)
	for irect in range(rects.shape[0]):
		rect = rects[irect]
		point = pnts[irect]
		cv2.rectangle(img, (int(rect[0]), int(rect[1])), (int(rect[2]), int(rect[3])), (255,0,0), 2)
		for ip in range(5):
			cv2.circle(img, (int(point[ip]), int(point[ip+5])), 2, (0,255,0))
	cv2.imshow("img", img)
	cv2.waitKey()
