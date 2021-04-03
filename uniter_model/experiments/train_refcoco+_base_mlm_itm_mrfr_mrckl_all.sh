# bert-base with all-tasks pre-trained on all data (COCO+VG+CC+SBU)

GPU=$1

# pre-trained model
checkpoint=/pretrain/bert-base_weak_alldata_2nodes_16accum/ckpt/model_step_125000.pt

# parameters
warmup=1000
steps=10000
lr=5e-5
batch_size=128
gradient_accumulation_steps=1
mlp=1

# output name
output_name=bert-base_allweak_alldata-refcoco+_w${warmup}_s${steps}_l${lr}_b${batch_size}_g${gradient_accumulation_steps}_m${mlp}
echo ${output_name}

CUDA_VISIBLE_DEVICES=${GPU} horovodrun -np 1 -H localhost:1 \
    python train_re.py \
        --train_txt_db /db/refcoco+_train_base-cased.db \
        --train_img_dir /img/visual_grounding_coco_gt \
        --val_txt_db /db/refcoco+_val_base-cased.db \
        --val_img_dir /img/visual_grounding_det_coco \
        --checkpoint ${checkpoint} \
        --cut_bert -1 \
        --output_dir /storage/refcoco+/${output_name} \
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