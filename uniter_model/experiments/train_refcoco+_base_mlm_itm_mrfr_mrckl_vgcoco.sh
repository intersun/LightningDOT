
# bert-base with all-tasks pre-trained on COCO+VG 
checkpoint=/pretrain/ablation/mlm_mrfr_mrckl_itm.pt
output_name=bert-base_mlm_itm_mrfr_mrckl_itm_pretrain_cocovg-refcoco+_12k_mlp1

horovodrun -np 1 -H localhost:1 \
    python train_re.py \
        --train_txt_db /db/refcoco+_train_base-cased.db \
        --train_img_dir /img/visual_grounding_coco_gt \
        --val_txt_db /db/refcoco+_val_base-cased.db \
        --val_img_dir /img/visual_grounding_det_coco \
        --checkpoint ${checkpoint} \
        --cut_bert -1 \
        --output_dir /storage/refcoco+/${output_name} \
        --max_txt_len 60 \
        --train_batch_size 128 \
        --val_batch_size 128 \
        --learning_rate 8e-5 \
        --optim adamw \
        --betas 0.9 0.98 \
        --weight_decay 0.01 \
        --dropout 0.1 \
        --grad_norm 2.0 \
        --decay linear \
        --num_train_steps 12000 \
        --warmup_steps 1500 \
        --gradient_accumulation_steps 1 \
        --seed 24 \
        --mlp 1 \
        --fp16

#########################
# This one is even better
#########################
# ablation_pretrained_model=mlm_mrfr_mrckl_itm
# checkpoint=/pretrain/ablation/mlm_mrfr_mrckl_itm.pt;
# output_name=bert-base_mlm_itm_mrfr_mrckl_itm_pretrain_cocovg-refcoco+_step10k

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
#         --num_train_steps 10000 \
#         --warmup_steps 1500 \
#         --gradient_accumulation_steps 1 \
#         --seed 24 \
#         --mlp 1 \
#         --fp16

