FROM python:3.8.8-slim-buster

RUN apt-get update
RUN apt-get install -y python-pip
RUN pip install matplotlib tqdm pycocotools opencv-python
RUN apt-get install -y libgl1-mesa-glx
RUN pip install torch torchvision

WORKDIR /workspace
