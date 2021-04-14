# Docker build file for MagellanMapper with a Venv environment

FROM ubuntu:16.04

ENV BASE_DIR /app

# install wget, apt-transport-https, and gnupg for AdoptOpenJDK; libsm6
# and libgl1-mesa-glx fof VTK; and vim for any basic text editing
RUN apt-get update \
    && apt-get install -y \
        wget \
        sudo \
        vim \
        apt-transport-https \
        gnupg \
        libsm6 \
        libgl1-mesa-glx

# install AdoptOpenJDK and Python 3.6 from extra repos
RUN echo "deb http://ppa.launchpad.net/deadsnakes/ppa/ubuntu xenial main" \
        >> /etc/apt/sources.list \
    && echo "deb-src http://ppa.launchpad.net/deadsnakes/ppa/ubuntu xenial main" \
        >> /etc/apt/sources.list \
    && apt-key adv --keyserver keyserver.ubuntu.com --recv-keys \
        F23C5A6CF475977595C89F51BA6932366A755776 \
    && wget -qO - https://adoptopenjdk.jfrog.io/adoptopenjdk/api/gpg/key/public \
        | sudo apt-key add - \
    && echo "deb https://adoptopenjdk.jfrog.io/adoptopenjdk/deb xenial main" \
        | sudo tee /etc/apt/sources.list.d/adoptopenjdk.list \
    && apt-get update \
    && apt-get install -y \
        python3.6 \
        python3.6-venv \
        python3.6-dev \
        adoptopenjdk-15-hotspot-jre \
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

# set up appliction base directory and change to non-root user
WORKDIR $BASE_DIR
USER $username

# set up Venv environment for MagellanMapper
COPY --chown=$username:$username bin/setup_venv.sh bin/libmag.sh ./bin/
COPY --chown=$username:$username setup.py ./
RUN bin/setup_venv.sh -e /home/$username/venvs \
    && rm -rf /home/$username/.cache/pip \
    && echo "export JAVA_HOME=/usr/lib/jvm/adoptopenjdk-15-hotspot-jre-amd64" \
        >> ~/.bashrc \
    && echo ". /home/$username/venvs/mag/bin/activate" >> ~/.bashrc \
    && . /home/$username/venvs/mag/bin/activate

# extract application contents from a git archive to use only files in
# the repository; copy after Venv setup to avoid triggering rebuilding
# prior layers for code updates
COPY --chown=$username:$username magellanmapper_gitarc.tar.gz ./
RUN tar xzvf magellanmapper_gitarc.tar.gz && rm magellanmapper_gitarc.tar.gz
