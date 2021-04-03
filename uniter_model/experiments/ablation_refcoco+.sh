# Supports ablation study of the follows:
# 1)  scratch
# 2)  bert
# 3)  mrfr
# 4)  mlm
# 5)  itm
# 6)  mlm_itm
# 7)  mlm_mrfr_itm
# 8)  mlm_mrc_itm
# 9)  mlm_mrckl_itm
# 10) mlm_mrfr_mrc_itm
# 11) mlm_mrfr_mrckl_itm
# 12) mlm_mrfr_mrckl_itm_jrm
# 13) mlm_mrfr_mrckl_itm_jrm+

ablation_pretrained_model=$1

case $ablation_pretrained_model in
    scratch|bert|mrfr|mlm|itm|mlm_itm|mlm_mrfr_itm|mlm_mrc_itm|mlm_mrckl_itm|mlm_mrfr_mrc_itm|mlm_mrfr_mrckl_itm|mlm_mrfr_mrckl_itm_jrm|mlm_mrfr_mrckl_itm_jrm+)
        echo running $ablation_pretrained_model ...;;
    *)
        echo "$ablation_pretrained_model" not supported.;
        exit 1;
esac

if [ "$ablation_pretrained_model" == "mrfr" ]; then
    cut_bert=1
else
    cut_bert=-1
fi

case $ablation_pretrained_model in
    scratch)
        cut_bert=1;
        checkpoint="scratch";;
    bert)
        cut_bert=1;
        checkpoint="google-bert";;
    mrfr)
        cut_bert=1;
        checkpoint=/pretrain/ablation/"${ablation_pretrained_model}".pt;;
    *)
        cut_bert=-1;
        checkpoint=/pretrain/ablation/"${ablation_pretrained_model}".pt;;
esac

horovodrun -np 1 -H localhost:1 \
    python train_re.py \
        --train_txt_db /db/refcoco+_train_base-cased.db \
        --train_img_dir /img/visual_grounding_coco_gt \
        --val_txt_db /db/refcoco+_val_base-cased.db \
        --val_img_dir /img/visual_grounding_det_coco \
        --checkpoint ${checkpoint} \
        --cut_bert ${cut_bert} \
        --output_dir /storage/refcoco+/ablation_"${ablation_pretrained_model}" \
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
        --num_train_steps 24000 \
        --warmup_steps 1500 \
        --gradient_accumulation_steps 1 \
        --seed 24 \
        --mlp 1 \
        --fp16
