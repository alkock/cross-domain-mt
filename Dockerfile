ARG BASE_IMAGE="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime"
FROM ${BASE_IMAGE}

EXPOSE 8888 6006

ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ca-certificates \
        git \
        build-essential \
        cmake \
        libboost-all-dev \
        libgomp1 \
        libopenblas-base \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace


COPY requirements.txt ./

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y screen unzip wget vim libgl1 libglib2.0-0

RUN pip install --no-cache-dir --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]
