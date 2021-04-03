# # bert-base with all-tasks pre-trained on CC+SBU

# checkpoint=/pretrain/bert-base_weak_conceptual_sbu_mlm_mrm_itm_mrckl_3xV100_run2/ckpt/model_step_100000.pt
# output_name=bert-base_mlm_itm_mrfr_mrckl_itm_pretrain_ccsbu-refcoco+_12k_mlp2

# horovodrun -np 1 -H localhost:1 \
#     python train_re.py \
#         --train_txt_db /db/refcoco+_train_base-cased.db \
#         --train_img_dir /img/visual_grounding_coco_gt \
#         --val_txt_db /db/refcoco+_val_base-cased.db \
#         --val_img_dir /img/visual_grounding_det_coco \
#         --checkpoint ${checkpoint} \
#         --cut_bert -1 \
#         --output_dir /storage/refcoco+/${output_name} \
#         --max_txt_len 60 \
#         --train_batch_size 128 \
#         --val_batch_size 128 \
#         --learning_rate 8e-5 \
#         --optim adamw \
#         --betas 0.9 0.98 \
#         --weight_decay 0.01 \
#         --dropout 0.1 \
#         --grad_norm 2.0 \
#         --decay linear \
#         --num_train_steps 12000 \
#         --warmup_steps 1500 \
#         --gradient_accumulation_steps 1 \
#         --seed 24 \
#         --mlp 2 \
#         --fp16

GPU=$1

# pre-trained model
checkpoint=/pretrain/bert-base_weak_conceptual_sbu_mlm_mrm_itm_mrckl_3xV100_run2/ckpt/model_step_100000.pt

# parameters
warmup=1000
steps=12000
lr=6e-5
batch_size=64
gradient_accumulation_steps=1
mlp=2

# output name
output_name=bert-base_allweak_ccsbu-refcoco+_w${warmup}_s${steps}_l${lr}_b${batch_size}_g${gradient_accumulation_steps}_m${mlp}
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