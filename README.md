# NNStreamer Examples (fork by Aaron Arthurs)

This fork demonstrates object-detection in a sample video using TF-Lite and NNStreamer.

## Installation:
- `git clone https://github.com/ajarthurs/nnstreamer-example.git <SOURCEDIR>`
- `mkdir <WORKSPACE>`
- `cd <WORKSPACE>`
- `cp -a <SOURCEDIR>/Dockerfile .`
- `cp -a <SOURCEDIR>/docker-compose.yml .`
- `cp -a <SOURCEDIR>/docker_entrypoint.bash .`
- `docker-compose build`

## Setup VNC connection:
- (Recommended. Setup an SSH tunnel to `localhost:5900`).
- `cd <WORKSPACE>`
- `docker-compose up`
- Open a VNC client and connect to `localhost:5900`.

## Run demo:
- At the VNC client, type the following inside the Xterm window.
- `cd /usr/local/bin`
- `sh get-model-object-detection-tflite.sh`
- `GST_DEBUG_DUMP_DOT_DIR=/tmp ./nnstreamer_example_object_detection_tflite`

## (Optional. Convert pipeline DOT output).
- `dot -Tpdf /tmp/pipeline.dot > /demo/pipeline.pdf`
- Note that in the Docker image, `/demo` is mounted to `<WORKSPACE>` on the host. On the host, you may access `<WORKSPACE>/pipeline.pdf`.

## Shutdown:
- Close Xterm window, which will shutdown the session.
- Back at the host terminal, run `docker-compose down --remove-orphans`


# ORIGINAL README.md:
This repository shows developers how to create their applications with nnstreamer/gstreamer. We recommend to install nnstreamer by downloading prebuilt binary packages from Launchpad/PPA (Ubuntu) or Download.Tizen.org (Tizen). If you want to build nnstreamer in your system for your example application builds, pdebuild (Ubuntu) with PPA or gbs (Tizen) are recommended for building nnstreamer. This repository has been detached from nnstreamer.git to build examples independently from the nnstreamer souce code since Jan-09-2019.

Ubuntu PPA: nnstreamer/ppa [[PPA Main](https://launchpad.net/~nnstreamer/+archive/ubuntu/ppa)]<br />
Tizen Project: devel:AIC:Tizen:5.0:nnsuite [[OBS Project](https://build.tizen.org/project/show/devel:AIC:Tizen:5.0:nnsuite)] [[RPM Repo](http://download.tizen.org/live/devel%3A/AIC%3A/Tizen%3A/5.0%3A/nnsuite/standard/)]


We provide example nnstreamer applications:

- Traditional Linux native applications
   - Linux/Ubuntu: GTK+ application
   - gst-launch-1.0 based scripts
- Tizen GUI Application
   - Tizen C/C++ application
   - Tizen .NET (C#) application
   - Tizen Web application
- Android applications
   - NDK based C/C++ CLI applicaton
   - JNI based GUI application
