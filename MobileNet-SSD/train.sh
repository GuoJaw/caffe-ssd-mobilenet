#!/bin/sh

mkdir -p snapshot
../build/tools/caffe train -solver="solver_train.prototxt" \  #已经改好
-weights="MobileNet_Pre_Train.caffemodel" \ #已经改好
-gpu 0 
