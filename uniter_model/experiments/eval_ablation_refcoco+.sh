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

horovodrun -np 1 -H localhost:1 \
    python eval_re.py \
        --txt_db /db/refcoco+_val_base-cased.db:/db/refcoco+_testA_base-cased.db:/db/refcoco+_testB_base-cased.db \
        --img_dir /img/visual_grounding_coco_gt \
        --output_dir /storage/refcoco+/ablation_${ablation_pretrained_model} \
        --checkpoint best

horovodrun -np 1 -H localhost:1 \
    python eval_re.py \
        --txt_db /db/refcoco+_val_base-cased.db:/db/refcoco+_testA_base-cased.db:/db/refcoco+_testB_base-cased.db \
        --img_dir /img/visual_grounding_det_coco \
        --output_dir /storage/refcoco+/ablation_${ablation_pretrained_model}  \
        --checkpoint best
