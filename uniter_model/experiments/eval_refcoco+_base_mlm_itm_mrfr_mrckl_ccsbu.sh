# det val: 72.49, testA: 79.20, testB: 63.22; gt val: 80.23, testA: 83.71, testB: 75.62 
# output_name=bert-base_allweak_ccsbu-refcoco+_w1000_s10000_l5e-5_b128_g1_m1

# det val: 72.63, testA: 79.18, testB: 63.37; gt val: 80.37, testA: 83.30, testB: 75.31
# output_name=bert-base_allweak_ccsbu-refcoco+_w1200_s12000_l8e-5_b128_g1_m1


# # det val: 72.63, testA: 78.83, testB: 63.76; gt val: 80.45, testA: 83.58, testB: 75.82 
# GPU=$1
# output_name=bert-base_allweak_ccsbu-refcoco+_w1000_s10000_l6e-5_b128_g1_m1

# det val: 72.90, testA: 79.01, testB: 63.53; gt val: 80.82, testA: 83.65, testB: 76.46 
GPU=$1
output_name=bert-base_allweak_ccsbu-refcoco+_w1000_s12000_l6e-5_b64_g1_m2

# print
echo $output_name

CUDA_VISIBLE_DEVICES=${GPU} horovodrun -np 1 -H localhost:1 \
    python eval_re.py \
        --txt_db /db/refcoco+_val_base-cased.db:/db/refcoco+_testA_base-cased.db:/db/refcoco+_testB_base-cased.db \
        --img_dir /img/visual_grounding_coco_gt \
        --output_dir /storage/refcoco+/${output_name} \
        --checkpoint best

CUDA_VISIBLE_DEVICES=${GPU} horovodrun -np 1 -H localhost:1 \
    python eval_re.py \
        --txt_db /db/refcoco+_val_base-cased.db:/db/refcoco+_testA_base-cased.db:/db/refcoco+_testB_base-cased.db \
        --img_dir /img/visual_grounding_det_coco \
        --output_dir /storage/refcoco+/${output_name} \
        --checkpoint best


