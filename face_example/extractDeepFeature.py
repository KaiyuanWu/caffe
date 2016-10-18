# coding: utf-8
import caffe
from caffe.proto import caffe_pb2
from find_face import FindFace
import  cv2
import mxnet as mx
import numpy as np
import importlib

def resdual_block(data, layerid, blockid, num_filter):
    conv1 = mx.symbol.Convolution(name="conv%d_%d"%(layerid, 2*blockid-1), data=data, kernel=(3,3), num_filter=num_filter, pad=(1, 1))
    relu1 = mx.symbol.LeakyReLU(name= "relu%d_%d"%(layerid, 2*blockid-1) , data=conv1, act_type='prelu')
    conv2 = mx.symbol.Convolution(name="conv%d_%d"%(layerid, 2*blockid), data=relu1, kernel=(3,3), num_filter=num_filter, pad=(1, 1))
    relu2 = mx.symbol.LeakyReLU(name="relu%d_%d"%(layerid, 2*blockid), data=conv2, act_type='prelu')
    res = data + relu2
    return res

def layer_block(data, layerid, num_block, num_filter, is_exit_layer=False):
    for i in range(num_block):
        data = resdual_block(data, layerid=layerid, blockid=i+1, num_filter = num_filter)
    if is_exit_layer:
        return data
    else:
        conv = mx.symbol.Convolution(name="conv%d"%(layerid), data=data, kernel=(3, 3), num_filter=2*num_filter)
        relu = mx.symbol.LeakyReLU(name="relu%d"%(layerid), data=conv, act_type='prelu')
        pool = mx.symbol.Pooling(name="pool%d"%(layerid), data=relu, pool_type="max", kernel=(2, 2), stride=(2, 2))
        return pool

def get_network():
    #Entry Flow
    data = mx.symbol.Variable(name="data")
    conv1a = mx.symbol.Convolution(name='conv1a', data=data, kernel=(3, 3), num_filter=32)
    relu1a = mx.symbol.LeakyReLU(name="relu1a", data=conv1a, act_type='prelu')
    conv1b = mx.symbol.Convolution(name='conv1b', data=relu1a, kernel=(3, 3), num_filter=64)
    relu1b = mx.symbol.LeakyReLU(name="relu1b", data=conv1b, act_type='prelu')
    pool1b = mx.symbol.Pooling(name='pool1b', data=relu1b, pool_type="max", kernel=(2, 2), stride=(2, 2))

    #Middle Flow
    layer2 = layer_block(data = pool1b, layerid = 2, num_block = 1, num_filter = 64)
    layer3 = layer_block(data=layer2, layerid=3, num_block=2, num_filter=128)
    layer4 = layer_block(data=layer3, layerid=4, num_block=5, num_filter=256)
    layer5 = layer_block(data=layer4, layerid=5, num_block=3, num_filter=512, is_exit_layer = True)

    #Exit Flow
    fc5 = mx.symbol.FullyConnected(name = "fc5", data=layer5, num_hidden=512)
    return fc5

def copy_caffemodel(caffe_prototxt, caffe_modelfile):
    net = caffe.Net(caffe_prototxt, caffe_modelfile, caffe.TEST)
    arg_params = {}
    for name in net.params:
        if name.find('conv') != -1 or name.find('fc') != -1:
            arg_params[name+"_weight"] = net.params[name][0].data.copy()
            arg_params[name + "_bias"] = net.params[name][1].data.copy()
        else:
            if name.find('relu') != -1:
                arg_params[name+'_gamma'] = net.params[name][0].data.copy()
            else:
                print("Unknown layer: "+name)
                quit()
    return arg_params

#从cp2tfrom.m 中 findSimilarity 翻译过来的
def findNonreflectiveSimilarity(xy, uv):
    npnts = xy.shape[0]
    X = np.zeros((2*npnts, 4))
    X[:npnts,:2] = xy[:2,:]
    X[:npnts,2] = 1
    X[npnts:,0] = xy[1,:]
    X[npnts:,1] = -xy[0, :]
    X[:npnts,3] = 1
    U = uv.flatten()
    r,err,_,_ = np.linalg.lstsq(X, U)
    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    T = np.array([[sc, -ss, 0],
                  [ss, sc, 0],
                  [tx, ty, 1]])
    return T,err

def crop_face(img, rect):
    imgSize = (112, 96)
    coord5points = [30.2946, 65.5318, 48.0252, 33.5493, 62.7299, 51.6963, 51.5014, 71.7366, 92.3655, 92.2041]
    xy = np.array([rect[:5], rect[5:]])
    T1, err1 = findNonreflectiveSimilarity(xy = xy, uv= coord5points)
    xy = np.array([rect[:5], rect[5:]])
    T2, err2 = findNonreflectiveSimilarity(xy = xy, uv= coord5points)
    if err2 < err1:
        T = T1
    else:
        T= T2
    face = cv2.warpAffine(img = img, M = T, dsize = imgSize)
    return face

#network = get_network()
#a = mx.viz.plot_network(network, shape={"data":(1, 1, 112, 96)}, node_attrs={"shape":'rect',"fixedsize":'false'})
#a.render("net")



input_shape = {'data':(1, 3, 112, 96)}
network = get_network()
executor = network.simple_bind(ctx = mx.cpu(), **input_shape)

#初始化参数
caffe.set_mode_cpu()
caffe_modelprefix = '/Users/kaiwu/Documents/veryes/FaceVerification/caffe-face/face_example/'
arg_params = copy_caffemodel(caffe_modelprefix + 'face_deploy.prototxt', caffe_modelprefix + 'face_model.caffemodel')
for arg in executor.arg_dict:
    if arg != 'data':
        executor.arg_dict[arg][:] = arg_params[arg]


#人脸检测器和标志点标定
model_prefix = '/Users/kaiwu/Documents/veryes/FaceAlignments/MTCNN/MTCNN_face_detection_alignment/code/codes/MTCNNv2/model/'
PNet = {"deploy_proto": model_prefix + "det1.prototxt",
		"model": model_prefix + "det1.caffemodel"}
RNet = {"deploy_proto": model_prefix + "det2.prototxt",
		"model": model_prefix + "det2.caffemodel"}
ONet = {"deploy_proto": model_prefix + "det3.prototxt",
		"model": model_prefix + "det3.caffemodel"}
LNet = {"deploy_proto": model_prefix + "det4.prototxt",
		"model": model_prefix + "det4.caffemodel"}

face = FindFace(PNet, RNet, ONet, LNet)

imgfile = '/Users/kaiwu/Documents/veryes/FaceVerification/caffe-face/face_example/Jennifer_Aniston_0016.jpg'
img = cv2.imread(imgfile)
minl = min(img.shape[0], img.shape[1])
minsize = int(minl * 0.1)
rects, pnts = face.detect_face(img, minsize)


