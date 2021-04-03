horovodrun -np 1 -H localhost:1 \
    python eval_re.py \
        --txt_db /db/refcoco+_val_large-cased.db:/db/refcoco+_testA_large-cased.db:/db/refcoco+_testB_large-cased.db \
        --img_dir /img/visual_grounding_coco_gt \
        --output_dir /storage/refcoco+/bert-large_mlm+itm+mrfr+mrckl_pretrain_alldata-refcoco+_lr8e-5_2mlp \
        --checkpoint 40

horovodrun -np 1 -H localhost:1 \
    python eval_re.py \
        --txt_db /db/refcoco+_val_large-cased.db:/db/refcoco+_testA_large-cased.db:/db/refcoco+_testB_large-cased.db \
        --img_dir /img/visual_grounding_det_coco \
        --output_dir /storage/refcoco+/bert-large_mlm+itm+mrfr+mrckl_pretrain_alldata-refcoco+_lr8e-5_2mlp \
        --checkpoint 40

