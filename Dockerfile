
# 1. SETUP BASE IMAGE
# --------
FROM nvidia/cuda:10.2-base-ubuntu18.04
# search in nvidia dockerhub


# 2. SETUP BASE LINUX REQUIREMENTS
# --------
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends python3.8 python3-pip python3-setuptools && \
    apt-get clean
# "--no-install-recommends" to avoid not-needed dependencies and create a light image
# "apt-get clean" to clean up cache and reduce image size


# 3. SETUP PYTHON REQUIREMENTS
# --------
WORKDIR /opt/bronchinet

COPY ["./requirements.txt", "./setup.py", "./"]
# copy files from local dir (this repository) to inside the container
#   destination path relative to WORKDIR

RUN /usr/bin/python3 -m pip install --upgrade pip && \
    pip3 install -r "./requirements.txt"
# "/usr/bin/python3 -m pip install --upgrade pip" to solve issues with tensorflow package


# 4. PREPARE WORKSPACE
# --------
ENV PYTHONPATH "/opt/bronchinet/src"

COPY ["./src/", "./src/"]

ARG MODELDIR=./models/
# used-defined variable in docker build (--build-arg MODELDIR=<desired_value>)
#   default value "./models/"

WORKDIR /workdir

COPY ["${MODELDIR}", "./models/"]
# destination path now relative to new WORKDIR="/workdir/"

RUN ln -s "/opt/bronchinet/src" "./Code"


# 5. FINALISE
# --------
# comment-out only for debugging
#RUN apt-get install -y vim
#ENTRYPOINT ["/bin/bash"]

ENTRYPOINT ["/bin/bash", "./models/run_trained_model.sh"]

#CMD ["./output_results/", "--testing_datadir=./inputdata/"]
# used-defined variables in docker run
