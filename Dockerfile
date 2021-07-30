
# 1. Setup base image
FROM nvidia/cuda:10.2-base-ubuntu18.04

# 2. Setup base linux requirements
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends python3.8 python3-pip python3-setuptools && \
    apt-get clean

# 3. Setup python requirements
WORKDIR /opt/bronchinet
# the docker daemon is in the root of this repository. The builder is in "/opt/bronchinet"
COPY ["./requirements.txt", "./setup.py", "./"]

RUN /usr/bin/python3 -m pip install --upgrade pip && \
    pip3 install -r "./requirements.txt"

# 4. Prepare workspace
ENV PYTHONPATH "/opt/bronchinet/src"

COPY ["./src/", "./src/"]

# input argument in docker build (--build-arg MODELDIR=<desired_input>)
ARG MODELDIR=./models/

WORKDIR /workdir

COPY ["${MODELDIR}", "./models/"]

RUN mkdir "./output_results/"

ENTRYPOINT ["/bin/bash"]

#ENTRYPOINT ["python3", "/opt/bronchinet/src/scripts_launch/launch_predictions_full.py"]

#CMD ["./models/model_evalEXACT.pt", "./output_results/", "--testing_datadir=./inputdata/"]
