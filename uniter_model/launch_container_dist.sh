TXT_DB=$1
IMG_DIR=$2
OUTPUT=$3
PRETRAIN_DIR=$4

if [ -z $CUDA_VISIBLE_DEVICES ]; then
    CUDA_VISIBLE_DEVICES='all'
fi

if [ "$5" = "--prepro" ]; then
    RO=""
else
    RO=",readonly"
fi

sudo docker run --gpus "device=$CUDA_VISIBLE_DEVICES" --ipc=host --rm -it \
    --privileged=true \
    --network=host \
    -v /convaistorage3mmb0/horovod_cluster_keys:/root/.ssh \
    -v /convaistorage3mmb0:/convaistorage3mmb0 \
    -v /convaistorage3mmb1:/convaistorage3mmb1 \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$OUTPUT,dst=/storage,type=bind \
    --mount src=$PRETRAIN_DIR,dst=/pretrain,type=bind,readonly \
    --mount src=$TXT_DB,dst=/db,type=bind$RO \
    --mount src=$IMG_DIR,dst=/img,type=bind,readonly \
    -e NCCL_IB_CUDA_SUPPORT=0 \
    -w /src convaicontainerregistry1.azurecr.io/img-txt
