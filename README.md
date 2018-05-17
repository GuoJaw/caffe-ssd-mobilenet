Caffe-SSD-Mobilenet 和 Caffe-SSD
（1）环境：
	和编译Caffe一样
测试环境是否成功搭建：python2 demo.py
cd caffe-ssd-mobile
python2 demo.py

（2）MobileNet-SSD文件夹，其中重要文件简介如下：


	
（2）模型训练：
1.建立数据集软连接
用SSD创建生成的LMDB文件，然后建立软连接
	$ cd ~/MobileNet-SSD
	$ ln ‐s /home/gjw/data/KITTIdevkit/KITTI/lmdb/KITTI_trainval_lmdb trainval_lmdb
	$ ln ‐s /home/gjw/data/KITTIdevkit/KITTI/lmdb/KITTI_test_lmdb test_lmdb
执行完命令，就能在项目文件夹下看到trainval_lmdb和test_lmdb软连接。

2.创建labelmap.prototxt文件
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


3.运行gen_model.sh脚本：用template中的prototxt，生成example中的prototxt（生成的prototxt是已经合并过bn层的）
	./gen_model.sh 类别数+1 ， 即./gen_model.sh 4
执行之后，得到examples文件夹，里面的3个prototxt。

4.修改训练和测试超参数
	修改solver_train.prototxt和solver_test.prototxt。
	test_iter=测试集图片数量/batchsize；
	初始学习率不宜太高，否则基础权重破坏比较严重；
	优化算法是RMSProp，可能对收敛有好处，不要改成SGD，也是为了保护权重。

5.预训练模型（代码中已经存在）：MobileNet_Pre_Train.caffemodel

6.开始训练,测试网络精度
修改并运行train.sh脚本，中途可以不断调节参数。
训练结束后，运行test.sh脚本，测试网络的精度值。

============================================

7.合并bn层,生成检测模型
为了提高模型运行速度，作者在这里将bn层合并到了卷积层中，相当于bn的计算时间就被节省了，对检测速度可能有小幅度的帮助，打开merge_bn.py文件，然后注意修改其中的文件路径：

caffe_root = '/home/gjw/caffe-ssd-mobile/'
train_proto = 'example/MobileNetSSD_train.prototxt'   #训练使用的example/MobileNetSSD_train.prototxt
train_model = 'MobileNetSSD_train.caffemodel'  # 训练生成的caffemodel路径
deploy_proto = 'example/MobileNetSSD_deploy.prototxt'  #训练使用的example/MobileNetSSD_deploy.prototxt
save_model = 'MobileNetSSD_deploy.caffemodel'  #合并后，caffemodel的保存路径

         然后运行该脚本，就可以得到最终的检测模型，那这个模型由于合并了bn层，参数格式已经变化，就不能再用于训练了。
如果想继续训练，应该用合并前的。


8.depthwise convolution layer加速
	epthwise_conv_layer.hpp
	depthwise_conv_layer.cpp
	depthwise_conv_layer.cu

(1)修改MobileNetSSD_deploy.prototxt，将其中所有名为convXX/dw（XX代指数字）的type从”Convolution”替换成”DepthwiseConvolution”，总共需要替换13处，从conv1/dw到conv13/dw
(2)把“engine: CAFFE”都注释掉，这个新的网络文件可以另存为MobileNetSSD_deploy_depth.prototxt
注释：caffemodel模型不用动，只需要指定新的prototxt文件和含有depthwise convolution layer的Caffe即可。




