## 相关代码
	#coding=utf-8    ## 中文注释

	import numpy as np
	import sys,os
	import cv2
	import caffe  
	import time

	##设置caffe_root，开启gpu模式
	caffe_root = '/home/gjw/caffe-ssd-mobile/'
	sys.path.insert(0, caffe_root + 'python')  
	caffe.set_mode_gpu()   ### 设置GPU模式

	if __name__ == "__main__":   ## python主函数
	    detect()
    
## Intro
	包含Caffe-SSD-Mobilenet 和 Caffe-SSD 和 Classification

## 环境搭建：

	和编译Caffe一样
	
	caffe.proto 生成caffe.pb.cc： caffe.pb.cc / caffe.pb.h，拷贝到相应位置（百度下即可）
	protoc src/caffe/proto/caffe.proto --cpp_out=.
	mkdir include/caffe/proto
	mv src/caffe/proto/caffe.pb.h include/caffe/proto
	
## 测试环境是否成功搭建：python2 demo.py

	cd caffe-ssd-mobile
	python2 demo.py

	
## Caffe-SSD-Mobilenet 模型训练：

## 1.建立数据集软连接

	用SSD创建生成的LMDB文件，然后建立软连接
		$ cd ~/MobileNet-SSD
		$ ln ‐s /home/gjw/data/KITTIdevkit/KITTI/lmdb/KITTI_trainval_lmdb trainval_lmdb
		$ ln ‐s /home/gjw/data/KITTIdevkit/KITTI/lmdb/KITTI_test_lmdb test_lmdb
	执行完命令，就能在项目文件夹下看到trainval_lmdb和test_lmdb软连接。

## 2.创建labelmap.prototxt文件
	item {
	  name: "none_of_the_above"
	  label: 0
	  display_name: "background"
	  }
	  item {
	  name: "Car"
	  label: 1
	  display_name: "Car"
	  }
	  item {
	  name: "Pedestrian"
	  label: 2
	  display_name: "Pedestrian"
	  }
	  item {
	  name: "Cyclist"
	  label: 3
	  display_name: "Cyclist"
	  }


## 3.运行gen_model.sh脚本：用template中的prototxt，生成example中的prototxt（生成的prototxt是已经合并过bn层的）

	./gen_model.sh 类别数+1 ， 即./gen_model.sh 4
	执行之后，得到examples文件夹，里面的3个prototxt。

## 4.修改训练和测试超参数

	修改solver_train.prototxt和solver_test.prototxt。
	test_iter=测试集图片数量/batchsize；
	初始学习率不宜太高，否则基础权重破坏比较严重；
	优化算法是RMSProp，可能对收敛有好处，不要改成SGD，也是为了保护权重。

## 5.预训练模型（代码中已经存在）：MobileNet_Pre_Train.caffemodel

## 6.开始训练,测试网络精度

	修改并运行train.sh脚本，中途可以不断调节参数。
	训练结束后，运行test.sh脚本，测试网络的精度值。

## ============================================

## 7.合并bn层,生成检测模型

	为了提高模型运行速度，作者在这里将bn层合并到了卷积层中，相当于bn的计算时间就被节省了，对检测速度可能有小幅度的帮助，打开merge_bn.py文件，然后注意修改其中的文件路径：

	caffe_root = '/home/gjw/caffe-ssd-mobile/'
	train_proto = 'example/MobileNetSSD_train.prototxt'   #训练使用的example/MobileNetSSD_train.prototxt
	train_model = 'MobileNetSSD_train.caffemodel'  # 训练生成的caffemodel路径
	deploy_proto = 'example/MobileNetSSD_deploy.prototxt'  #训练使用的example/MobileNetSSD_deploy.prototxt
	save_model = 'MobileNetSSD_deploy.caffemodel'  #合并后，caffemodel的保存路径

		 然后运行该脚本，就可以得到最终的检测模型，那这个模型由于合并了bn层，参数格式已经变化，就不能再用于训练了。
	如果想继续训练，应该用合并前的。

## 8.depthwise convolution layer加速

	epthwise_conv_layer.hpp
	depthwise_conv_layer.cpp
	depthwise_conv_layer.cu

	(1)修改MobileNetSSD_deploy.prototxt，将其中所有名为convXX/dw（XX代指数字）的type从”Convolution”替换成”DepthwiseConvolution”，总共需要替换13处，从conv1/dw到conv13/dw
	(2)把“engine: CAFFE”都注释掉，这个新的网络文件可以另存为MobileNetSSD_deploy_depth.prototxt
	注释：caffemodel模型不用动，只需要指定新的prototxt文件和含有depthwise convolution layer的Caffe即可。


## Classification  二分类（Caffe+Alexnet）

## 一、训练数据集的准备

	（1）将数据集放在data/EyeData目录下
	数据集目录结构：
	~/data
		    EyeData
			    train
				    open
				    close
			    val
				    open
				    close
	说明：
		open和close分别存放不同类别的图像

	（2）新建/home/gjw/caffe-ssd/data/EyeData目录，“软连接”train和val数据集
	gjw@gjw:~/caffe-ssd/data/EyeData$ ln -s ~/data/EyeData/train train
	gjw@gjw:~/caffe-ssd/data/EyeData$ ln -s ~/data/EyeData/val  val


	（3）制作数据集脚本，生成train.txt和val.txt
	import os  

	pwd_dir = os.getcwd() 
	data = 'train'  
	path = os.listdir(pwd_dir+'/'+ data) 
	path.sort()  
	file = open('train.txt','w')  

	i = 0  

	for line in path:  
	  str = pwd_dir+'/'+ data +'/'+line    #  /pwd/train/
	  for child in os.listdir(str):  
	    str1 = data+'/'+line+'/'+child;  
	    d = ' %s' %(i)  
	    t = str1 + d  
	    file.write(t +'\n')  
	  i=i+1  

	file.close() 


	执行 python2 train_txt.py，生成train.txt
	执行 python2 val_txt.py，生成val.txt

## +++++++++++++++++++++++++++++++++++++++++++++++++

## 二、生成leveldb数据集、均值文件

	【1】生成leveldb格式的数据集
	./build/tools/convert_imageset --resize_height=256 --resize_width=256 --shuffle ./data/EyeData/  ./data/EyeData/train.txt  ./data/EyeData/eye_train_leveldb --backend=leveldb

	./build/tools/convert_imageset --resize_height=256 --resize_width=256 --shuffle ./data/EyeData/  ./data/EyeData/val.txt  ./data/EyeData/eye_val_leveldb --backend=leveldb

	【2】生成均值文件
	./build/tools/compute_image_mean  ./data/EyeData/eye_train_leveldb  ./data/EyeData/mean.binaryproto --backend=leveldb

## +++++++++++++++++++++++++++++++++++++++++++++++++

## 三、网络配置文件（将bvlc_alexnet文件拷贝到caffe-ssd/data/EyeData目录下），修改网络配置文件

	（三/一）train_val.prototxt，修改见下：
	【1】均值文件，leveldb文件，batch_size
	name: "AlexNet"
	layer {
	  name: "data"
	  type: "Data"
	  top: "data"
	  top: "label"
	  include {
	    phase: TRAIN
	  }
	  transform_param {
	    mirror: true
	    crop_size: 227
	    mean_file: "data/EyeData/mean.binaryproto"  ###
	  }
	  data_param {
	    source: "data/EyeData/eye_train_leveldb"  ###
	    batch_size: 32  ###256
	    backend: LEVELDB  ###
	  }
	}
	layer {
	  name: "data"
	  type: "Data"
	  top: "data"
	  top: "label"
	  include {
	    phase: TEST
	  }
	  transform_param {
	    mirror: false
	    crop_size: 227
	    mean_file: "data/EyeData/mean.binaryproto"  ###
	  }
	  data_param {
	    source: "data/EyeData/eye_val_leveldb"   ####
	    batch_size: 5   ####50
	    backend: LEVELDB   ###
	  }
	}

	【2】最后一层fc8（重点）
	【2-1】修改num_output为自己的类别数2
	【2-2】fc8层修改
	    修改fc8名字为fc8eye    ####必须改名字
		lr_mult: 10  ##1
		decay_mult: 10   ##1

		     lr_mult: 20   ## 2
	修改见下：

	layer {
	  name: "fc8eye"
	  type: "InnerProduct"
	  bottom: "fc7"
	  top: "fc8eye"
	  param {
	    lr_mult: 10  ##1
	    decay_mult: 10   ##1
	  }
	  param {
	    lr_mult: 20   ## 2
	    decay_mult: 0
	  }
	  inner_product_param {
	    num_output: 2   ###1000
	    weight_filler {
	      type: "gaussian"
	      std: 0.01
	    }
	    bias_filler {
	      type: "constant"
	      value: 0
	    }
	  }
	}

	（三/二）slover.prototxt修改
	net: "data/EyeData/bvlc_alexnet/train_val.prototxt"  ##train_val.prototxt文件所在目录
	test_iter: 10  ###1000
	test_interval: 500   ###1000
	base_lr: 0.001   ### 学习率 0.01
	lr_policy: "step"
	gamma: 0.1
	stepsize: 5000   ###100000
	display: 20   ##  多少次显示一次
	max_iter: 1500    ###  最大迭代次数，最重要的参数----->坑死爹了，训练次数过多，过拟合
	momentum: 0.9
	weight_decay: 0.0005
	snapshot: 10000    ###多少次保存一次模型
	snapshot_prefix: "data/EyeData/bvlc_alexnet/model/alexnet"  ## 模型保存路径model；模型名字alexnet_iter_1500.caffemodel
	solver_mode: GPU

## +++++++++++++++++++++++++++++++++++++++++++++++++   

## （四）训练
	./build/tools/caffe train --solver=data/EyeData/bvlc_alexnet/solver.prototxt -gpu 0

## +++++++++++++++++++++++++++++++++++++++++++++++++

## （五）进行测试单张图像类别预测
	【1】deploy.prototxt修改（用于：预测图片class）：只修改fc8层，修改与train_val.prototxt的fc8层必须一致
	layer {
	  name: "fc8eye"
	  type: "InnerProduct"
	  bottom: "fc7"
	  top: "fc8eye"
	  param {
	    lr_mult: 10  ##1
	    decay_mult: 10   ##1
	  }
	  param {
	    lr_mult: 20   ## 2
	    decay_mult: 0
	  }
	  inner_product_param {
	    num_output: 2   ###1000
	    weight_filler {
	      type: "gaussian"
	      std: 0.01
	    }
	    bias_filler {
	      type: "constant"
	      value: 0
	    }
	  }
	}

	【2】执行下面命令进行单张测试
	gjw@gjw:~/caffe-ssd$ 
		./build/examples/cpp_classification/classification.bin data/EyeData/bvlc_alexnet/deploy.prototxt  data/EyeData/bvlc_alexnet/model/alexnet_iter_10000.caffemodel data/EyeData/mean.binaryproto data/EyeData/bvlc_alexnet/labels.txt  data/EyeData/test/0.jpg  








