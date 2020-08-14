# Docker build file for compiling SimpleElastix
# Provides binary compatibility for older Linux kernels and Glibc
# Author: David Young, 2020

FROM ubuntu:16.04

# run with login Bash shell to allow Conda init
SHELL ["/bin/bash", "--login", "-c"]
ENV BASE_DIR /app

# installer build tools
RUN  apt-get update && apt-get install -y \
    wget \
    git \
    gcc \
    gawk \
    bison \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/*

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

# copy in custom SimpleElastix build and Conda setup scripts
COPY --chown=$username:$username bin/setup_conda bin/libmag.sh bin/build_se.sh \
    ./bin/

# download and extract CMake version required for SimpleElastix
RUN wget https://cmake.org/files/v3.10/cmake-3.10.3-Linux-x86_64.tar.gz \
    && tar xzvf cmake-3.10.3-Linux-x86_64.tar.gz \
    && echo "export PATH=\"/$BASE_DIR/cmake-3.10.3-Linux-x86_64/bin:$PATH\"" \
    >> ~/.bashrc

# set up basic Conda environment with Python 3.6 and build SimpleElastix
# within this environment
RUN export PATH="$BASE_DIR/cmake-3.10.3-Linux-x86_64/bin:$PATH" \
    && echo -e "dependencies:\n  - python=3.6" > env.yml \
    && echo "y" | bin/setup_conda -n sd -s env.yml \
    && echo "conda activate sd" >> ~/.bashrc \
    && eval "$(/home/"$username"/miniconda3/bin/conda shell.bash hook)" \
    && conda clean --all \
    && rm -rf /home/$username/.cache/pip \
    && rm Miniconda3-latest-Linux-x86_64.sh \
    && conda activate sd \
    && bin/build_se.sh -s SimpleElastix -d build_se
