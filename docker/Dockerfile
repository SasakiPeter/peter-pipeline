FROM python:3.7.4
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN apt-get update && apt-get install -y \
  sudo \
  cmake \
  time \
  graphviz \
  xdg-utils \
  python3-setuptools &&\
  apt-get clean && \ 
  rm -rf /var/lib/apt/lists/* &&\
  pip --no-cache-dir install --upgrade pip && \
  pip --no-cache-dir install -r requirements.txt
COPY . /code/