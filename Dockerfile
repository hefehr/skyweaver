FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

LABEL maintainer="erc-compact"

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update && \
    apt-get install -yq --no-install-recommends \
    cmake \
    doxygen \
    git \
    graphviz \
    libboost-all-dev \
    libtool \
    python-is-python3 \
    python3-sphinx 

### PYTHON DEPENDENCIES
RUN apt-get install -yq --no-install-recommends \
    python3-pip \
    python3-setuptools \
    python3-wheel

RUN pip3 install --upgrade pip && \
    pip3 install numpy scipy matplotlib

### PSRDADA
RUN cd /usr/src && \
    git clone https://git.code.sf.net/p/psrdada/code psrdada && \
    cd psrdada && \
    git checkout bf756866898686065562ac537376cf9d7d1b54ee && \
    # Version from  Wed Feb 12 07:46:37 2020 +0200 \
    export LDFLAGS=-fPIC && \
    export CFLAGS=-fPIC && \
    ./bootstrap && \
    ./configure --includedir=/usr/local/include/psrdada --with-cuda-dir=/usr/local/cuda/ && \
    make -j 8 && \
    make install

### PSRDADA_CPP  \
RUN cd /usr/src && \
    git clone https://gitlab.mpcdf.mpg.de/mpifr-bdg/psrdada_cpp.git && \
    cd psrdada_cpp/ &&\
    git checkout a3853de61ef56ab3cb3df23e954de24444dfe0e3 && \
    cmake -S . -B build/ -DENABLE_CUDA=True \
          -DARCH=native -DPSRDADA_INCLUDE_DIR=/usr/local/include/psrdada \
          -DBUILD_SUBMODULES=OFF .. &&\
    make -C build/ -j8 && make -C build/ install

### SKYWEAVER
WORKDIR /usr/src/skyweaver
COPY . . 
#RUN  cmake -S . -B build/ -DARCH=native \
#        -DPSRDADA_INCLUDE_DIR=/usr/local/include/psrdada \
#        -DPSRDADACPP_INCLUDE_DIR=/usr/local/include/psrdada_cpp \
#        -DBUILD_SUBMODULES=OFF .. &&\
#        make -C build/ -j8 && make -C build/ install

### Install nsight-systems profiler
RUN apt update && \
    apt install -y --no-install-recommends gnupg && \
    echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub && \
    apt update && \
    apt install nsight-systems-cli