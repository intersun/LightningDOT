# output_name=bert-large_mlm+itm+mrfr+mrckl_pretrain_alldata-refcoco+_lr8e-5_2mlp
# output_name=bert-large_allweak_alldata-refcoco+_w1000_s10000_l5e-5_b64_g2_m1

# det val: 74.94, testA: 81.24, testB: 65.06.
# gd  val: 84.04, testA: 85.87, testB: 78.89;
# output_name=bert-large_allweak_alldata-refcoco+_w1000_s10000_l5e-5_b64_g2_m2

# 
output_name=bert-large_allweak_alldata-refcoco+_w1000_s10000_l5e-5_b32_g2_m2

echo ${output_name}

horovodrun -np 1 -H localhost:1 \
    python eval_re.py \
        --txt_db /db/refcoco+_val_large-cased.db:/db/refcoco+_testA_large-cased.db:/db/refcoco+_testB_large-cased.db \
        --img_dir /img/visual_grounding_coco_gt \
        --output_dir /storage/refcoco+/${output_name} \
        --checkpoint best

horovodrun -np 1 -H localhost:1 \
    python eval_re.py \
        --txt_db /db/refcoco+_val_large-cased.db:/db/refcoco+_testA_large-cased.db:/db/refcoco+_testB_large-cased.db \
        --img_dir /img/visual_grounding_det_coco \
        --output_dir /storage/refcoco+/${output_name} \
        --checkpoint best

