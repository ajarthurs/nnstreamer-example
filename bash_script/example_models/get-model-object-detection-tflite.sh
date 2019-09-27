#!/usr/bin/env bash

# Sample video from Coverr@pexels.com showing Broadway Street (16 sec).
wget 'https://www.dropbox.com/s/v2knofo4z19ggue/sample_1080p.mp4?dl=0' -O sample_1080p.mp4

# Models
mkdir -p tflite_model
cd tflite_model

# COCO
wget  https://github.com/nnsuite/testcases/raw/master/DeepLearningModels/tensorflow-lite/ssd_mobilenet_v2_coco/coco_labels_list.txt

# SSD-MobileNetV1 COCO
wget 'https://www.dropbox.com/s/l2ryee8ikbx1as1/float32.tflite?dl=0' -O ssd_mobilenet_v1_coco.tflite
wget  https://github.com/nnsuite/testcases/raw/master/DeepLearningModels/tensorflow-lite/ssd_mobilenet_v2_coco/box_priors.txt -O box_priors-ssd_mobilenet.txt
# SSD-FPN COCO
wget  https://www.dropbox.com/s/02wh04g9oo9s5vg/float32.tflite?dl=0 -O ssd_resnet50_v1_fpn_coco.tflite
wget  https://www.dropbox.com/s/jtwt7tjr2bc0dh5/float32.tflite?dl=0 -O ssd_mobilenet_v1_fpn_coco.tflite
wget  https://www.dropbox.com/s/fmrbtmdcp6sqz3a/box_priors.txt?dl=0 -O box_priors-ssd_fpn.txt
