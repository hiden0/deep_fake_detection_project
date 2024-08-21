#IMPORTANT!!!! 
# Be carefull with the image version, if gpu drivers will be updated, this
# tensorflow gpu image could fail. Run test_gpu.py inside docker to check the
# compatibility with cuda version installed in the container and the gpu drivers.
#FROM tensorflow/tensorflow:2.14.0-gpu    
#FROM tensorflow/tensorflow:latest-gpu-jupyter
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
#FROM nvidia/cuda-ppc64le:10.2-cudnn8-devel
ARG USER_ID
ARG GROUP_ID
ARG USER_NAME


EXPOSE 6006

ENV TENSORBOARD_PORT=6006


# To avoid using the root user in the container, we create a user/group with 
# the same IDs than in the host
RUN if [ ${USER_ID:-0} -ne 0 ] && [ ${GROUP_ID:-0} -ne 0 ]; then \
        if id ${USER_NAME}; then userdel -f ${USER_NAME}; fi && \ 
        if getent group ${USER_NAME} ; then groupdel ${USER_NAME}; fi && \
        groupadd -g ${GROUP_ID} ${USER_NAME} && \
        useradd -m -l -u ${USER_ID} -g ${USER_NAME} ${USER_NAME} ; \
    else echo "Error with argument values. You must give values to USER_ID, GROUP_ID and USER_NAME using --build-arg." && exit 1 \
    ;fi
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get update && apt-get install -y screen \
    wget 
#RUN apt install python3 -y
#RUN apt install python3-pip -y

RUN mkdir -p /opt/conda 
COPY miniconda.sh  /tmp
RUN bash tmp/miniconda.sh -b -u -p ~/miniconda3
# Install your environment.yaml deps into base env 
# Uncomment once you are ready to start productionizing the image 
#COPY environment_cuda11.8.yml /tmp 

#RUN . /opt/miniconda/bin/activate && conda env update --name base --file /tmp/environment.yaml 




USER ${USER_NAME}

ENV SHELL /bin/bash

SHELL ["/bin/bash", "-c"]

#ENV PATH="$PATH:/home/${USER_NAME}/.local/bin"

#WORKDIR /repositorio


#COPY ./docker/requirements_docker.txt /tmp

#RUN pip install --no-cache-dir -r /tmp/requirements_docker.txt



COPY ./entrypoint.sh /tmp/entrypoint.sh
#RUN pip install --no-cache-dir -r /tmp/requirements_docker.txt
#ENTRYPOINT ["/bin/bash"]
WORKDIR ${WORKDIR}
#RUN chown -R ${USER_NAME}:${USER_NAME} ${WORKDIR}

ENTRYPOINT ["/bin/bash", "/tmp/entrypoint.sh"]

#ENTRYPOINT ["/bin/bash", "/repositorio/Docker/entrypoint.sh"]