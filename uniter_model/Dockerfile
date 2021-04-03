FROM nvcr.io/nvidia/pytorch:19.05-py3
COPY requirements.txt scripts/download_bert.py ./
RUN pip install -r requirements.txt &&\
    python download_bert.py &&\
    rm ./requirements.txt ./download_bert.py

################## v1 ##########################

COPY scripts/install_horovod.sh ./
RUN source install_horovod.sh &&\
    rm ./install_horovod.sh
ENV OPENMPI_VERSION=4.0.0

# fix ssh permissions
RUN bash -c "chmod -R 600 /etc/ssh/ && chmod 600 /var/run/sshd/ && chmod 600 /root"

################## horovod, v2 ##########################


RUN bash -c "pip install lz4==2.1.9 lmdb==0.97"

################# LMDB ##########################
