horovodrun -np 1 -H localhost:1 \
    python eval_re.py \
        --txt_db /db/refcoco+_val_base-cased.db:/db/refcoco+_testA_base-cased.db:/db/refcoco+_testB_base-cased.db \
        --img_dir /img/visual_grounding_coco_gt \
        --output_dir /storage/refcoco+/bert-base_mlm+itm+mrfr_pretrain-refcoco+_lr1e-4 \
        --checkpoint 26 

horovodrun -np 1 -H localhost:1 \
    python eval_re.py \
        --txt_db /db/refcoco+_val_base-cased.db:/db/refcoco+_testA_base-cased.db:/db/refcoco+_testB_base-cased.db \
        --img_dir /img/visual_grounding_det_coco \
        --output_dir /storage/refcoco+/bert-base_mlm+itm+mrfr_pretrain-refcoco+_lr1e-4 \
        --checkpoint 26
