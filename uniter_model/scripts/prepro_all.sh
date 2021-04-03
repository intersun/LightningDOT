TOKER=$1
TXT_DB=$2
FORMAT=$3
#TXT_DB='/ssd2/yenchun/TXT_DB_test'

ANNOTATIONS='/ssd2/yenchun/ANNOTATIONS'
VQA_ANN=$ANNOTATIONS/VQA/
CAP_ANN=$ANNOTATIONS/COCO_annotation/
CONCEPT_ANN=$ANNOTATIONS/conceptual_captions/
SBU_ANN=$ANNOTATIONS/sbu_caption/
PRETRAIN_ANN=$ANNOTATIONS/latest_cleaned/
ITM_ANN=$ANNOTATIONS/Image-Text-Matching
VE_ANN=$ANNOTATIONS/visual_entailment/
GQA_ANN=$ANNOTATIONS/GQA/
VCR_ANN=$ANNOTATIONS/VCR/
NLVR2_ANN=$ANNOTATIONS/NLVR2/


# process licheng's split
#python scripts/split_annotations.py --format $FORMAT \
#    $PRETRAIN_ANN/collected\(coco+vg\).json $PRETRAIN_ANN


if [ $TOKER = 'bert-large-cased' ]; then
    SUFFIX='large-cased'
elif [ $TOKER = 'bert-base-cased' ]; then
    SUFFIX='base-cased'
else
    echo "invalid tokenizer specified"
    exit(1)
fi

# Image Text Retrieval
for DSET in 'flickr30k' 'coco'; do
    for SPLIT in 'train' 'val' 'test'; do
        python prepro.py --task itm --bert $TOKER --format $FORMAT \
            --annotations $ITM_ANN/${DSET}_$SPLIT.json \
            --output $TXT_DB/itm_${DSET}_${SPLIT}_$SUFFIX.db

    done
done
# coco 1k splits
for SPLIT in 'val' 'test'; do
    for i in 0 1 2 3 4; do
        python prepro.py --task itm --bert $TOKER --format $FORMAT \
            --annotations $ITM_ANN/coco_${SPLIT}_1k_$i.json \
            --output $TXT_DB/itm_coco_${SPLIT}_1k_${i}_$SUFFIX.db
    done
done
# coco val rest
python prepro.py --task itm --bert $TOKER --format $FORMAT \
    --annotations $ITM_ANN/coco_restval.json \
    --output $TXT_DB/itm_coco_restval_$SUFFIX.db


# COCO
for SPLIT in 'train' 'val'; do
    # VQA
    python prepro.py --task vqa --bert $TOKER --format $FORMAT \
        --annotations $VQA_ANN/v2_OpenEnded_mscoco_${SPLIT}2014_questions.json \
            $VQA_ANN/v2_mscoco_${SPLIT}2014_annotations.json \
            $VQA_ANN/ans2label.pkl \
        --output $TXT_DB/vqa_${SPLIT}_$SUFFIX.db
    if [ $SPLIT = 'val' ]; then
        for SP in 'train' 'dev'; do
            python prepro.py --task vqa --bert $TOKER --format $FORMAT \
                --annotations $VQA_ANN/v2_OpenEnded_mscoco_${SP}val2014_questions.json \
                    $VQA_ANN/v2_mscoco_${SP}val2014_annotations.json \
                    $VQA_ANN/ans2label.pkl \
                --output $TXT_DB/vqa_${SP}val_$SUFFIX.db
        done
    fi

    # Caption
    python prepro.py --task caption --bert $TOKER --format $FORMAT \
        --annotations $CAP_ANN/captions_${SPLIT}2014.json \
        --output $TXT_DB/caption_${SPLIT}_$SUFFIX.db
done

# COCO VQA test
python prepro.py --task vqa --bert $TOKER --format $FORMAT \
    --annotations $VQA_ANN/v2_OpenEnded_mscoco_test2015_questions.json \
    --output $TXT_DB/vqa_test_$SUFFIX.db

# VG VQA
python prepro.py --task vqa --bert $TOKER --format $FORMAT \
    --annotations $VQA_ANN/VG_questions.json.mapped \
        $VQA_ANN/VG_annotations.json.mapped \
        $VQA_ANN/ans2label.pkl \
    --output $TXT_DB/vqa_vg_$SUFFIX.db

# all pretraining

# coco trainval
python prepro.py --task licheng_cleaned --bert $TOKER --format $FORMAT \
    --annotations $PRETRAIN_ANN/pretrain_caption_coco_trainval.json \
    --output $TXT_DB/pretrain_caption_coco_trainval_$SUFFIX.db

for DSET in 'coco' 'vg'; do
    for SPLIT in 'val' 'train'; do
        python prepro.py --task licheng_cleaned --bert $TOKER --format $FORMAT \
            --annotations $PRETRAIN_ANN/pretrain_caption_${DSET}_$SPLIT.json \
            --output $TXT_DB/pretrain_caption_${DSET}_${SPLIT}_$SUFFIX.db
    done
done

# pretrain VQA
for DSET in 'genome_vqa' 'gqa'; do
    if [ $DSET = 'genome_vqa' ]; then
        DS='vg'
    else
        DS='gqa'
    fi
    for SPLIT in 'val' 'train'; do
        python prepro.py --task vqa --bert $TOKER --format $FORMAT \
            --annotations $PRETRAIN_ANN/${DSET}_${SPLIT}_questions.json \
                $PRETRAIN_ANN/${DSET}_${SPLIT}_annotations.json \
                $PRETRAIN_ANN/ans2label.pkl \
            --output $TXT_DB/pretrain_vqa_${DS}_${SPLIT}_$SUFFIX.db
    done
done
# Pretrain VQA COCO
for SPLIT in 'val' 'trainsplit' 'valsplit' ; do
    python prepro.py --task vqa --bert $TOKER --format $FORMAT \
        --annotations $PRETRAIN_ANN/coco_vqa_${SPLIT}_questions.json \
            $PRETRAIN_ANN/coco_vqa_${SPLIT}_annotations.json \
            $PRETRAIN_ANN/ans2label.pkl \
        --output $TXT_DB/pretrain_vqa_coco_${SPLIT}_$SUFFIX.db
done


# Visual Entailment
for SPLIT in 'train' 'dev' 'test'; do
    python prepro.py --task ve --bert $TOKER --format $FORMAT \
        --annotations $VE_ANN/snli_ve_$SPLIT.jsonl \
        --output $TXT_DB/ve_${SPLIT}_$SUFFIX.db
done

# GQA
for SPLIT in 'train' 'val' 'testdev'; do
    for VER in 'all' 'balanced'; do
        python prepro.py --task vqa --bert $TOKER --format $FORMAT \
            --annotations $GQA_ANN/gqa_${SPLIT}_${VER}_questions.vqa.json \
                $GQA_ANN/gqa_${SPLIT}_${VER}_annotations.vqa.json \
                $GQA_ANN/ans2label.pkl \
            --output $TXT_DB/gqa_${SPLIT}_${VER}_$SUFFIX.db
    done
done
# GQA test
python prepro.py --task vqa --bert $TOKER --format $FORMAT \
    --annotations $GQA_ANN/gqa_submission_questions.vqa.json \
    --output $TXT_DB/gqa_submission_$SUFFIX.db


# Conceptual Captions
for SPLIT in 'train' 'val'; do
    python prepro.py --task conceptual --bert $TOKER --format $FORMAT \
        --annotations $CONCEPT_ANN/${SPLIT}_imageId2Ann.tsv \
            $CONCEPT_ANN/${SPLIT}_imgs.json \
        --output $TXT_DB/conceptual_caption_${SPLIT}_$SUFFIX.db
done

# SBU captions
for SPLIT in 'train' 'val'; do
    python prepro.py --task sbu --bert $TOKER --format $FORMAT \
        --annotations $SBU_ANN/sbu_${SPLIT}_captions.json \
        --output $TXT_DB/sbu_caption_${SPLIT}_$SUFFIX.db
done

# VCR
for SPLIT in 'train' 'val'; do
    python prepro.py --task vcr --bert $TOKER --format $FORMAT \
        --annotations $VCR_ANN/$SPLIT.jsonl \
        --output $TXT_DB/vcr_${SPLIT}_$SUFFIX.db
done

# NLVR2
for SPLIT in 'dev' 'test1'; do
    python prepro.py --task nlvr2 --bert $TOKER --format $FORMAT \
        --annotations $NLVR2_ANN/$SPLIT.json \
        --output $TXT_DB/nlvr2_${SPLIT}_$SUFFIX.db
done
# some corrupted train features
python prepro.py --task nlvr2 --bert $TOKER --format $FORMAT \
    --annotations $NLVR2_ANN/train.json $NLVR2_ANN/train_imgs.json \
    --output $TXT_DB/nlvr2_train_$SUFFIX.db
