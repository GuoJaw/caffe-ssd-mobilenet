


#coding=utf-8

import numpy as np
import sys,os
import cv2
caffe_root = '/home/gjw/caffe-ssd-mobile/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  
import time

caffe.set_mode_gpu()   


net_file= 'deploy.prototxt'  
caffe_model='snapshot/_iter_10000.caffemodel' 

if not os.path.exists(caffe_model):
    print("caffemodel does not exist")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

CLASSES = ('background',
           'car', 'cyclist', 'pedestrian')

def preprocess(src):
    img = cv2.resize(src, (304,304))
    img_mean = np.array([103.94, 116.78, 123.68], dtype=np.float32)
    img = img.astype(np.float32, copy=True) - img_mean
    img = img * 0.017
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])
    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect_video():
    cap = cv2.VideoCapture("/home/gjw/caffe-ssd-mobile/kitti2.avi")  

    while (1) :
        ret, origimg = cap.read()
        if ret is False:
            print("Load video error")
            break
        start = time.time()

        img = preprocess(origimg)
        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))

        net.blobs['data'].data[...] = img
        out = net.forward()  
        box, conf, cls = postprocess(origimg, out)
        for i in range(len(box)):
           if conf[i] > 0.4 :
	      p1 = (box[i][0], box[i][1])
	      p2 = (box[i][2], box[i][3])
	      cv2.rectangle(origimg, p1, p2, (0,255,0))
	      p3 = (max(p1[0], 15), max(p1[1], 15))
	      title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
	      cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)

        end = time.time()  
        fps = 1 / (end - start);
        print('FPS = %.2f' % (fps))

        cv2.imshow("Pelee", origimg)
 
        k = cv2.waitKey(1) & 0xff
        if k == 27 : 
	    return False


if __name__ == '__main__':
    detect_video()



