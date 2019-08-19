# Builds a docker image that demonstrates TFLite NNStreamer object-detection.

FROM ubuntu:18.04
ENV DEBIAN_FRONTEND noninteractive
ENV DEBIAN_FRONTEND teletype

## Install APT utilities
RUN apt-get update && \
    apt-get upgrade -y
RUN apt-get install -y \
      software-properties-common \
      python3-software-properties
RUN apt-get install -y --no-install-recommends apt-utils

## Register and install fast APT sources. Needed to download packages faster than 10 KB/s.
RUN add-apt-repository ppa:apt-fast/stable && \
    apt-get update && \
    apt-get upgrade -y
RUN apt-get install -y apt-fast

## Register nnstreamer's repo
RUN add-apt-repository ppa:nnstreamer/ppa && \
    apt-get update && \
    apt-get upgrade -y

##XXX: Hack to enable man pages that are excluded by default
RUN sed --in-place 's@^path-exclude=/usr/share/man/@#&@' /etc/dpkg/dpkg.cfg.d/excludes
RUN apt-get --reinstall install -y man-db coreutils

## Install some useful supplemental tools
RUN apt-get install -y \
      git \
      graphviz \
      ssat \
      wget \
      vim

## Install nnstreamer and related
RUN apt-get install -y \
      meson \
      ninja-build \
      libgst-dev \
      libgstreamer1.0-dev \
      libgstreamer-plugins-base1.0-dev \
      libglib2.0-dev \
      libcairo2-dev \
      locales \
      gstreamer1.0-tools \
      gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad \
      gstreamer1.0-plugins-ugly \
      nnstreamer \
      nnstreamer-dev \
      nnstreamer-tensorflow \
      nnstreamer-tensorflow-lite \
      libprotobuf-dev \
      tensorflow-dev \
      tensorflow-lite-dev

## Install a VNC-accessible desktop environment. Needed to see video output from example.
RUN apt-get install -y \
  xorg \
  xterm \
  lxde-core \
  tigervnc-standalone-server

## Install codecs to support reading H.264 videos. This requires agreement to otherwise restricted codecs.
RUN ACCEPT_EULA=Y apt-get install -y ubuntu-restricted-extras

## Install nnstreamer demos using a fork of https://github.com/nnsuite/nnstreamer-example
RUN mkdir /src
RUN mkdir /demo
RUN git clone https://github.com/ajarthurs/nnstreamer-example.git /src/nnstreamer-example
WORKDIR /src/nnstreamer-example
RUN git checkout od-tflite-video-demo
RUN meson build && \
    ninja -C build install && \
    echo "DEMOS INSTALLED"
WORKDIR /demo
