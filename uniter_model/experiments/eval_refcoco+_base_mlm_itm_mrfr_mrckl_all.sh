# This training is done by experiments/train_refcoco+_base_mlm_itm_mrfr_mrckl_all.sh
# det val: 74.52, testA: 79.76, testB: 64.43
# gd  val: 82.74, testA: 85.21, testB: 77.52
horovodrun -np 1 -H localhost:1 \
    python eval_re.py \
        --txt_db /db/refcoco+_val_base-cased.db:/db/refcoco+_testA_base-cased.db:/db/refcoco+_testB_base-cased.db \
        --img_dir /img/visual_grounding_coco_gt \
        --output_dir /storage/refcoco+/bert-base_allweak_alldata-refcoco+_w1000_s10000_l5e-5_b128_g1_m1 \
        --checkpoint best

horovodrun -np 1 -H localhost:1 \
    python eval_re.py \
        --txt_db /db/refcoco+_val_base-cased.db:/db/refcoco+_testA_base-cased.db:/db/refcoco+_testB_base-cased.db \
        --img_dir /img/visual_grounding_det_coco \
        --output_dir /storage/refcoco+/bert-base_allweak_alldata-refcoco+_w1000_s10000_l5e-5_b128_g1_m1 \
        --checkpoint best

