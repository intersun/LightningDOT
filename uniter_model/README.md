# Universal-Image-Text-Transformer
Research code for pre-training universal vision and language models


## Requirements
nvidia driver (418.xx), docker(19.03+), nvidia-container-toolkit
```
docker pull convaicontainerregistry1.azurecr.io/img-txt
```

## lauching the environment
```
# can use CUDA_VISIBLE_DEVICES to seperate GPUs for each container
source launch_container.sh $TXT_DB $IMG_DIR $OUTPUT $PRETRAIN_PATH
# TXT_DB: convaistorage2share2/TXT_DB_v3
# IMG_DIR: convaistorage2share2/Bottom-up-features/adaptive/npy_per_img_id
# OUTPUT: somewhere to store model checkpoint (can be on share storage)
# PRETRAIN: path to pretrained model

# when need to preprocessing
source launch_container.sh $TXT_DB $IMG_DIR $OUTPUT $PRETRAIN_PATH --prepro
# this will make /db writable


# multi-node training
source launch_container_dist.sh $TXT_DB $IMG_DIR $OUTPUT $PRETRAIN_PATH
```

## Pretrain
```
# inside the docker container
horovodrun -np $N_GPU -H localhost:$N_GPU \
    python pretrain.py --config config/config-pretrain-alltask.json
```

## finetune VQA
```
horovodrun -np 2 -H localhost:2 \
    python train_vqa.py --config config/config-vqa-bert-2gpu-alldata.json
```
### VQA inference
```
# single node only
# please refer to code for commandline options
horovodrun -np $N_GPU -H localhost:$N_GPU \
    python eval_vqa.py --txt_db /db/vqa_test_[base/large]-cased.db/ \
        --img_dir /img/coco_test2015 --checkpoint [NUM] \
        --output_dir /path/to/trained/vqa
```

### NLVR2 official evaluation
Use official script to get both acc (our validation matched this) and consistency
```
# concat all output files
cat $OUTPUT/result/[val/test]_results_$STEP_rank*.csv > $OUTPUT.csv
python eval/nlvr2.py $OUTPUT.csv ANNOTATION.json
```

### Referring Expression Comprehension: Finetuning and Evaluation
```
# train on gd-truth pairs of (ref, sent)
horovodrun -np $N_GPU -H localhost:$N_GPU \
    python train_re.py --config config/hps-refcoco+.json

# evaluate multiple splits on gd-truth boxes
horovodrun -np $N_GPU -H localhost:$N_GPU \
    python eval_re.py \
        --txt_db /db/refcoco+_val_base-cased.db:/db/refcoco+_testA_base-cased.db:/db/refcoco+_testB_base-cased.db \
        --img_dir /img/visual_grounding_coco_gt \
        --output_dir /storage/refcoco+/bert-base_mlm+itm+mrfr_pretrain-refcoco+_lr1e-4 \
        --checkpoint 26

# evaluate multiple splits on detected boxes
horovodrun -np $N_GPU -H localhost:$N_GPU \
    python eval_re.py \
        --txt_db /db/refcoco+_val_base-cased.db:/db/refcoco+_testA_base-cased.db:/db/refcoco+_testB_base-cased.db \
        --img_dir /img/visual_grounding_det_coco \
        --output_dir /storage/refcoco+/bert-base_mlm+itm+mrfr_pretrain-refcoco+_lr1e-4 \
        --checkpoint 26
```

## Misc
1. w/o horovodrun it will run on single GPU
    - useful for debugger (-m pdb)
2. try `--pin_mem` it might give a tiny performance improvement
3. `--img_format [lmdb/lmdb-compress]`
    - trade-off between memory/CPU
    - use `--n_workers $N_CPU` to specify data workers (default: 4)

