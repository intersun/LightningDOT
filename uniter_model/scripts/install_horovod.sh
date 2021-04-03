# for building docker image

# Update OpenMPI to avoid bug
rm -r /usr/local/mpi

wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.0.tar.gz
gunzip -c openmpi-4.0.0.tar.gz | tar xf -
cd openmpi-4.0.0
./configure --prefix=/usr/local/mpi --enable-orterun-prefix-by-default \
    --disable-getpwuid
make -j$(nproc) all && make install
ldconfig

cd -
rm -r openmpi-4.0.0
rm openmpi-4.0.0.tar.gz

export OPENMPI_VERSION=4.0.0


# missing libnccl_static.a (solve by upgrading NCCL)
echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" \
    > /etc/apt/sources.list.d/nvidia-ml.list
apt update
apt install libnccl2=2.4.7-1+cuda10.1 libnccl-dev=2.4.7-1+cuda10.1

export PATH=/usr/local/mpi/bin:$PATH
HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_PYTORCH=1 \
    pip install --no-cache-dir horovod
ldconfig

# Install OpenSSH for MPI to communicate between containers
# apt-get install -y --no-install-recommends \
#    openssh-client openssh-server && \
# mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
# cat /etc/ssh/ssh_config | \
#    grep -v StrictHostKeyChecking > \
#    /etc/ssh/ssh_config.new && \
#    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
#    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config
