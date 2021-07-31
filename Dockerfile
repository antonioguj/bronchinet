
# 1. SETUP BASE IMAGE
# --------
FROM nvidia/cuda:10.2-base-ubuntu18.04
# base image where to start this docker (search in nvidia dockerhub)


# 2. SETUP BASE LINUX REQUIREMENTS
# --------
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends python3.8 python3-pip python3-setuptools && \
    apt-get clean
# "--no-install-recommends" to avoid not-needed dependencies and create lighter image
# "apt-get clean" to clean up cache memory and reduce image size


# 3. SETUP PYTHON REQUIREMENTS
# --------
WORKDIR /opt/bronchinet

COPY ["./requirements.txt", "./setup.py", "./"]
# copy files from local dir (this repository) to that inside the container
#   destination path relative to WORKDIR

RUN /usr/bin/python3 -m pip install --upgrade pip && \
    pip3 install -r "./requirements.txt"
# "/usr/bin/python3 -m pip install --upgrade pip" to solve issues with tensorflow package


# 4. PREPARE WORKSPACE
# --------
ENV PYTHONPATH "/opt/bronchinet/src/"

COPY ["./src/", "./src/"]

ARG MODELDIR=./models/
# used-defined variable to specify other paths for models
#   in docker build: --build-arg MODELDIR=<desired_value> (default "./models/")

WORKDIR /workdir

COPY ["${MODELDIR}", "./models/"]
# destination path now relative to WORKDIR="/workdir/"

RUN mkdir "./input_data/"
RUN mkdir "./results/"
# folders for input data / output results, to be mounted from local dirs in docker run

RUN ln -s "/opt/bronchinet/src/" "./Code"
RUN ln -s "./input_data/" "./BaseData"


# 5. FINALISE
# --------
ENTRYPOINT ["/bin/bash", "./models/run_model_trained.sh"]
# command to execute when running docker

CMD ["./input_data/", "./results/", "--docker"]
# input arguments to script in entrypoint

# FOR DEBUGGING: comment-out the commands above, and uncomment the ones below
#RUN apt-get install -y vim
#ENTRYPOINT ["/bin/bash"]