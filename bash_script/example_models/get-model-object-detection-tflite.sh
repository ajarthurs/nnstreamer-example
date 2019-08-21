#!/usr/bin/env bash

# Sample video from Coverr@pexels.com showing Broadway Street (16 sec).
wget 'https://gcs-vimeo.akamaized.net/exp=1566417848~acl=%2A%2F792677904.mp4%2A~hmac=b5e784c9702a2e2c98fc0fcddce6495e3549e77fcdada13b74a3ce379f698719/vimeo-prod-skyfire-std-us/01/159/9/225795843/792677904.mp4?download=1&filename=Pexels+Videos+4585.mp4' -O sample_1080p.mp4

# Model
mkdir -p tflite_model
cd tflite_model
wget https://github.com/nnsuite/testcases/raw/master/DeepLearningModels/tensorflow-lite/ssd_mobilenet_v2_coco/ssd_mobilenet_v2_coco.tflite
wget https://github.com/nnsuite/testcases/raw/master/DeepLearningModels/tensorflow-lite/ssd_mobilenet_v2_coco/coco_labels_list.txt
wget https://github.com/nnsuite/testcases/raw/master/DeepLearningModels/tensorflow-lite/ssd_mobilenet_v2_coco/box_priors.txt
