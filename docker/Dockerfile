FROM nvidia/cuda:9.2-devel
MAINTAINER Pradeep Kumar <pradeepisro49@gmail.com>

# Packages required to build ISPC and Clang.
RUN apt-get -y update && apt-get install -y wget build-essential vim gcc g++ git subversion python m4 bison flex zlib1g-dev ncurses-dev libtinfo-dev libc6-dev-i386 && \
    rm -rf /var/lib/apt/lists/*

# Download and install required version of cmake (3.8) for ISPC build
RUN wget https://cmake.org/files/v3.15/cmake-3.15.0-rc1-Linux-x86_64.sh && mkdir /opt/cmake && sh cmake-3.15.0-rc1-Linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake && rm cmake-3.15.0-rc1-Linux-x86_64.sh

# If you are behind a proxy, you need to configure git and svn.
#RUN git config --global --add http.proxy http://proxy.yourcompany.com:888

# Initialize svn configs
#RUN svn --version --quiet
#RUN echo "http-proxy-host=proxy.yourcompany.com" >> ~/.subversion/servers
#RUN echo "http-proxy-port=888" >> ~/.subversion/servers

WORKDIR /usr/local/src

# Fork ispc on github and clone *your* fork.
RUN git clone https://github.com/ispc/ispc.git

# This is home for Clang builds
RUN mkdir /usr/local/src/llvm

ENV ISPC_HOME=/usr/local/src/ispc
ENV LLVM_HOME=/usr/local/src/llvm

# If you are going to run test for future platforms, go to
# http://www.intel.com/software/sde and download the latest version,
# extract it, add to path and set SDE_HOME.

WORKDIR /usr/local/src/ispc

# Build Clang with all required patches.
# Pass required LLVM_VERSION with --build-arg LLVM_VERSION=<version>.
# By default 8.0 is used.
# Note self-build options, it's required to build clang and ispc with the same compiler,
# i.e. if clang was built by gcc, you may need to use gcc to build ispc (i.e. run "make gcc"),
# or better do clang selfbuild and use it for ispc build as well (i.e. just "make").
# "rm" are just to keep docker image small.
ARG LLVM_VERSION=8.0
RUN ./alloy.py -b --version=$LLVM_VERSION --selfbuild --git && \
    rm -rf $LLVM_HOME/build-$LLVM_VERSION $LLVM_HOME/llvm-$LLVM_VERSION $LLVM_HOME/bin-$LLVM_VERSION_temp $LLVM_HOME/build-$LLVM_VERSION_temp

ENV PATH=$LLVM_HOME/bin-$LLVM_VERSION/bin:$PATH

# Configure ISPC build
RUN mkdir build_$LLVM_VERSION
WORKDIR build_$LLVM_VERSION
RUN cmake ../ -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_INSTALL_PREFIX=/usr/local/src/ispc/bin-$LLVM_VERSION

# Build ISPC
RUN make ispc -j8 && make install
WORKDIR ../
RUN rm -rf build_$LLVM_VERSION

RUN echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial main" >> /etc/apt/sources.list && echo "deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial main" >> /etc/apt/sources.list && wget https://apt.llvm.org/llvm-snapshot.gpg.key && apt-key add llvm-snapshot.gpg.key

RUN apt update && apt-get install -y --no-install-recommends software-properties-common && add-apt-repository ppa:ubuntu-toolchain-r/test &&  apt update && apt install -y clang-9 libclang-9-dev llvm-9-tools libstdc++-9-dev

WORKDIR /usr/home/

RUN git clone https://github.com/schwarzschild-radius/spmdfy.git

WORKDIR /usr/home/spmdfy

RUN git checkout baseline && mkdir build

WORKDIR /usr/home/spmdfy/build

RUN cmake .. -DLLVM_DIR=/usr/lib/llvm-9/lib/cmake/llvm -DCMAKE_CXX_FLAGS=-std=c++17 && make
