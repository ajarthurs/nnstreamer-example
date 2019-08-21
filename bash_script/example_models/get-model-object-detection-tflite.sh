#!/usr/bin/env bash

# Sample video from Coverr@pexels.com showing Broadway Street (16 sec).
wget 'https://www.dropbox.com/s/v2knofo4z19ggue/sample_1080p.mp4?dl=0' -O sample_1080p.mp4

# Model
mkdir -p tflite_model
cd tflite_model
wget 'https://www.dropbox.com/s/l2ryee8ikbx1as1/float32.tflite?dl=0' -O ssd_mobilenet_v1_coco.tflite
wget https://github.com/nnsuite/testcases/raw/master/DeepLearningModels/tensorflow-lite/ssd_mobilenet_v2_coco/coco_labels_list.txt
wget https://github.com/nnsuite/testcases/raw/master/DeepLearningModels/tensorflow-lite/ssd_mobilenet_v2_coco/box_priors.txt
