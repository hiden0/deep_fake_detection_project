# Usando la imagen base con CUDA
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG USER_ID
ARG GROUP_ID
ARG USER_NAME

EXPOSE 6006

ENV TENSORBOARD_PORT=6006

# Crear usuario/grupo no root
RUN if [ ${USER_ID:-0} -ne 0 ] && [ ${GROUP_ID:-0} -ne 0 ]; then \
        if id ${USER_NAME}; then userdel -f ${USER_NAME}; fi && \ 
        if getent group ${USER_NAME} ; then groupdel ${USER_NAME}; fi && \
        groupadd -g ${GROUP_ID} ${USER_NAME} && \
        useradd -m -l -u ${USER_ID} -g ${USER_NAME} ${USER_NAME} ; \
    else echo "Error with argument values. You must give values to USER_ID, GROUP_ID and USER_NAME using --build-arg." && exit 1 \
    ;fi

# Instalar dependencias necesarias
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    screen \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Crear el directorio donde se instalará Miniconda
RUN mkdir -p /opt/conda

# Copiar el instalador de Miniconda al contenedor
COPY miniconda.sh /tmp/miniconda.sh

# Instalar Miniconda
RUN bash /tmp/miniconda.sh -b -u -p /opt/conda \
    && rm /tmp/miniconda.sh

# Añadir Miniconda al PATH
ENV PATH="/opt/conda/bin:$PATH"

# Cambiar al usuario no root
USER ${USER_NAME}

ENV SHELL /bin/bash
SHELL ["/bin/bash", "-c"]

# Copiar el script de entrada
COPY ./entrypoint.sh /tmp/entrypoint.sh

# Establecer el directorio de trabajo
WORKDIR /home/${USER_NAME}

# Cambiar los permisos del directorio de trabajo al usuario no root
RUN chown -R ${USER_NAME}:${USER_NAME} /home/${USER_NAME}

# Ejecutar el script de entrada
ENTRYPOINT ["/bin/bash", "/tmp/entrypoint.sh"]
