BERT=$1
PRETRAIN_DIR=$2


docker run --rm \
    --mount src=$(pwd),dst=/src,type=bind \
    --mount src=$PRETRAIN_DIR,dst=/pretrain,type=bind \
    convaicontainerregistry1.azurecr.io/uniter \
    python scripts/download_bert.py $BERT /pretrain/$BERT.pt
