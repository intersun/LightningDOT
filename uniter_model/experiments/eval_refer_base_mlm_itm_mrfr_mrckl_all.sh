# bert-base with all-tasks pre-trained on all data (COCO+VG+CC+SBU)

REFER=$1
GPU=$2

# parameters
warmup=1000
steps=12000 # 12000
lr=6e-5
batch_size=64 # 64
gradient_accumulation_steps=1
mlp=2

# output name
output_name=bert-base_allweak_alldata-${REFER}_w${warmup}_s${steps}_l${lr}_b${batch_size}_g${gradient_accumulation_steps}_m${mlp}
echo ${output_name}

# Evaluate
case ${REFER} in
    refcoco|refcoco+)
        dbs=/db/${REFER}_val_base-cased.db:/db/${REFER}_testA_base-cased.db:/db/${REFER}_testB_base-cased.db;;
    refcocog)
        dbs=/db/${REFER}_val_base-cased.db:/db/${REFER}_test_base-cased.db;;
esac

CUDA_VISIBLE_DEVICES=${GPU} horovodrun -np 1 -H localhost:1 \
    python eval_re.py \
        --txt_db ${dbs} \
        --img_dir /img/visual_grounding_coco_gt \
        --output_dir /storage/${REFER}/${output_name} \
        --checkpoint best

CUDA_VISIBLE_DEVICES=${GPU} horovodrun -np 1 -H localhost:1 \
    python eval_re.py \
        --txt_db ${dbs} \
        --img_dir /img/visual_grounding_det_coco \
        --output_dir /storage/${REFER}/${output_name} \
        --checkpoint best