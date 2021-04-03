
# bert-large with all-tasks pre-trained on all data (COCO+VG+CC+SBU)
REFER=$1
GPU=$2

# pre-trained model
checkpoint=/pretrain/bert-large_frkl_alldata.pt

# parameters
warmup=1000
steps=10000
lr=5.3e-5
batch_size=32
gradient_accumulation_steps=2
mlp=2

# output name
output_name=bert-large_allweak_alldata-${REFER}_w${warmup}_s${steps}_l${lr}_b${batch_size}_g${gradient_accumulation_steps}_m${mlp}
echo ${output_name}

CUDA_VISIBLE_DEVICES=${GPU} horovodrun -np 1 -H localhost:1 \
    python train_re.py \
        --train_txt_db /db/${REFER}_train_large-cased.db \
        --train_img_dir /img/visual_grounding_coco_gt \
        --val_txt_db /db/${REFER}_val_large-cased.db \
        --val_img_dir /img/visual_grounding_det_coco \
        --checkpoint ${checkpoint} \
        --cut_bert -1 \
        --output_dir /storage/${REFER}/${output_name} \
        --max_txt_len 60 \
        --train_batch_size ${batch_size} \
        --val_batch_size 128 \
        --learning_rate ${lr} \
        --optim adamw \
        --betas 0.9 0.98 \
        --weight_decay 0.01 \
        --dropout 0.1 \
        --grad_norm 2.0 \
        --decay linear \
        --num_train_steps ${steps} \
        --warmup_steps ${warmup} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --seed 24 \
        --mlp ${mlp} \
        --fp16

# Evaluate
case ${REFER} in
    refcoco|refcoco+)
        dbs=/db/${REFER}_val_large-cased.db:/db/${REFER}_testA_large-cased.db:/db/${REFER}_testB_large-cased.db;;
    refcocog)
        dbs=/db/${REFER}_val_large-cased.db:/db/${REFER}_test_large-cased.db;;
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
