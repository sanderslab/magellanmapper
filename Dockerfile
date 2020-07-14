# Docker build file for MagellanMapper
# Author: David Young, 2020

FROM continuumio/miniconda3

# run with login Bash shell to allow Conda init
SHELL ["/bin/bash", "--login", "-c"]
ENV BASE_DIR /app
RUN mkdir $BASE_DIR
WORKDIR $BASE_DIR
COPY environment.yml run.py setup.py LICENSE.txt ./
COPY magmap/ ./magmap/
COPY bin/ ./bin/
COPY stitch/ ./stitch/
COPY profiles/ ./profiles/

# create basic Conda env, initialize Conda, and set to activate env on login
#RUN conda create -n test01 python=3.6 && conda init bash \
#    && echo "conda activate test01" >> ~/.bashrc

RUN conda env create -n mag environment.yml && conda init bash \
    && echo "conda activate mag" >> ~/.bashrc
