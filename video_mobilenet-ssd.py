
#coding=utf-8

import numpy as np
import sys,os
import cv2
caffe_root = '/home/gjw/caffe-ssd-mobile/'
sys.path.insert(0, caffe_root + 'python')  
import caffe  
import time

caffe.set_mode_gpu()   ### 设置GPU模式


net_file= 'MobileNetSSD_deploy.prototxt'  
caffe_model='MobileNetSSD_deploy.caffemodel'  

if not os.path.exists(caffe_model):
    print("MobileNetSSD_deploy.caffemodel does not exist,")
    print("use merge_bn.py to generate it.")
    exit()
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')


def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect():

    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture('/home/gjw/ssd_sort-master/MOT06.mp4')  # 读取视频

    while (1) :
        ret, origimg = cap.read()
        if ret is False:
            break

        start = time.time()  # fps开始时间

        #origimg = cv2.imread(imgfile)
        img = preprocess(origimg)

        img = img.astype(np.float32)
        img = img.transpose((2, 0, 1))

        net.blobs['data'].data[...] = img

        start=time.time() # 开始时间

        out = net.forward()  #前向传播

        end = time.time()  # fps结束时间
        fps = 1 / (end - start);
        print('FPS = %.2f' % (fps))

        box, conf, cls = postprocess(origimg, out)

        for i in range(len(box)):
           p1 = (box[i][0], box[i][1])
           p2 = (box[i][2], box[i][3])
           cv2.rectangle(origimg, p1, p2, (0,255,0))
           p3 = (max(p1[0], 15), max(p1[1], 15))
           title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
           cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
        cv2.imshow("SSD", origimg)

        if(cv2.waitKey(1) & 0xff == 27):
                break

if __name__ == '__main__':
    detect()
