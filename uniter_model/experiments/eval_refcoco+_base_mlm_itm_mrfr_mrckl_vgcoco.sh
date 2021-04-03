# This training is done by experiments/ablation_refcoco+.sh mlm_mrfr_mrckl_itm
# det val: 74.52, testA: 79.76, testB: 64.43.
# gd  val: 82.74, testA: 85.21, testB: 77.52;
horovodrun -np 1 -H localhost:1 \
    python eval_re.py \
        --txt_db /db/refcoco+_val_base-cased.db:/db/refcoco+_testA_base-cased.db:/db/refcoco+_testB_base-cased.db \
        --img_dir /img/visual_grounding_coco_gt \
        --output_dir /storage/refcoco+/ablation_mlm_mrfr_mrckl_itm \
        --checkpoint best

horovodrun -np 1 -H localhost:1 \
    python eval_re.py \
        --txt_db /db/refcoco+_val_base-cased.db:/db/refcoco+_testA_base-cased.db:/db/refcoco+_testB_base-cased.db \
        --img_dir /img/visual_grounding_det_coco \
        --output_dir /storage/refcoco+/ablation_mlm_mrfr_mrckl_itm \
        --checkpoint best


# # This training is done by ./experiments/train_refcoco+_base_mlm_itm_mrfr_mrckl_vgcoco.sh
# # det val: 74.60, testA: 80.42, testB: 64.98
# # gd  val: 82.84, testA: 85.10, testB: 77.95
# horovodrun -np 1 -H localhost:1 \
#     python eval_re.py \
#         --txt_db /db/refcoco+_val_base-cased.db:/db/refcoco+_testA_base-cased.db:/db/refcoco+_testB_base-cased.db \
#         --img_dir /img/visual_grounding_coco_gt \
#         --output_dir /storage/refcoco+/bert-base_mlm_itm_mrfr_mrckl_itm_pretrain_cocovg-refcoco+_12k_mlp1 \
#         --checkpoint best

# horovodrun -np 1 -H localhost:1 \
#     python eval_re.py \
#         --txt_db /db/refcoco+_val_base-cased.db:/db/refcoco+_testA_base-cased.db:/db/refcoco+_testB_base-cased.db \
#         --img_dir /img/visual_grounding_det_coco \
#         --output_dir /storage/refcoco+/bert-base_mlm_itm_mrfr_mrckl_itm_pretrain_cocovg-refcoco+_12k_mlp1 \
#         --checkpoint best


