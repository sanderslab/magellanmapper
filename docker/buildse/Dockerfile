# Docker build file for compiling SimpleElastix
# Provides binary compatibility for older Linux kernels and Glibc
# Author: David Young, 2020

FROM ubuntu:16.04

ENV BASE_DIR /app

# install build tools
# - multiple Python version from deadsnakes PPA
# - build tools including specific CMake version required for SimpleElastix
# - OpenJDK 8 to build Python-Javabridge
# - vim to allow editing
RUN echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu xenial main" \
        >> /etc/apt/sources.list \
    && echo "deb-src http://ppa.launchpad.net/deadsnakes/ppa/ubuntu xenial main" \
        >> /etc/apt/sources.list \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys \
        F23C5A6CF475977595C89F51BA6932366A755776 \
    && apt-get update && apt-get install -y \
        wget \
        git \
        gcc \
        gawk \
        bison \
        make \
        g++ \
        vim \
        python3.6 \
        python3.6-venv \
        python3.6-dev \
        python3.7 \
        python3.7-venv \
        python3.7-dev \
        python3.8 \
        python3.8-venv \
        python3.8-dev \
        python3.9 \
        python3.9-venv \
        python3.9-dev \
        default-jdk \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoclean \
    && wget https://cmake.org/files/v3.10/cmake-3.10.3-Linux-x86_64.tar.gz -P /opt \
    && tar xzvf /opt/cmake-3.10.3-Linux-x86_64.tar.gz -C /opt \
    && rm /opt/cmake-3.10.3-Linux-x86_64.tar.gz

# set up non-root user with sudo access
ARG username=magellan
RUN mkdir /home/$username \
    && groupadd -r $username \
    && useradd -r -s /bin/false -g $username $username \
    && echo "$username:$username" | chpasswd \
    && usermod -aG sudo $username \
    && chown -R $username:$username /home/$username \
    && mkdir $BASE_DIR \
    && chown -R $username:$username $BASE_DIR

# set up appliction base diretory and change to non-root user
WORKDIR $BASE_DIR
USER $username

# copy in scripts for setting up multiple Venv environments
COPY --chown=$username:$username bin/libmag.sh bin/setup_multi_venvs.sh \
        ./bin/

# afterward, build SimpleElastix for multiple Python version by running,
# assuming the output parent directory `<out>` has been mounted and contains
# the SimpleElastix (otherwise it will be cloned there):
# `bin/build_deps.sh -d <out>/builds_se -e venvs -s <out>/SimpleElastix`
RUN echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' >> ~/.bashrc \
    && echo 'export PATH=/opt/cmake-3.10.3-Linux-x86_64/bin:$JAVA_HOME/bin:$PATH' \
        >> ~/.bashrc \
    && . ~/.bashrc \
    && bin/setup_multi_venvs.sh -d venvs

# copy in custom build scripts
COPY --chown=$username:$username bin/build_se.sh bin/build_jb.sh \
        bin/build_deps.sh ./bin/
