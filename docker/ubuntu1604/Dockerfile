# Docker build file for MagellanMapper
# Author: David Young, 2020

FROM ubuntu:16.04

# run with login Bash shell to allow Conda init
SHELL ["/bin/bash", "--login", "-c"]
ENV BASE_DIR /app
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

# install wget to download Miniconda
RUN  apt-get update && apt-get install -y wget git && rm -rf /var/lib/apt/lists/*
RUN  apt-get update && apt-get install -y wget git gcc gawk bison make \
    && rm -rf /var/lib/apt/lists/*

# set up appliction base diretory and change to non-root user
# set up appliction base diretory
WORKDIR $BASE_DIR

# build Glibc 2.27, required for SimpleITK
RUN wget https://ftp.gnu.org/gnu/glibc/glibc-2.27.tar.gz \
    && tar xzvf glibc-2.27.tar.gz && cd glibc-2.27 && mkdir build \
    && cd build && ../configure --prefix=/opt/glibc-2.27 \
    && make -j4 && make install && cd ../.. && rm -rf glibc-2.27 \
    && rm glibc-2.27.tar.gz

# change to non-root user
USER $username

# set up Conda environment for MagellanMapper
COPY --chown=$username:$username environment.yml ./
COPY --chown=$username:$username bin/setup_conda bin/libmag.sh ./bin/
RUN echo "y" | bin/setup_conda \
    && echo "conda activate mag" >> ~/.bashrc \
    && eval "$(/home/"$username"/miniconda3/bin/conda shell.bash hook)" \
    && conda clean --all \
    && rm -rf /home/$username/.cache/pip \
    && rm Miniconda3-latest-Linux-x86_64.sh

# extract application contents from a git archive to use only files in
# the repository; copy after Conda setup to avoid triggering rebuilding
# prior layers for code updates
COPY --chown=$username:$username magellanmapper_gitarc.tar.gz ./
RUN tar xzvf magellanmapper_gitarc.tar.gz \
    && rm magellanmapper_gitarc.tar.gz
