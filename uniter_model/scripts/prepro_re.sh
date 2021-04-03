TOKER='bert-base-cased'
TXT_DB='datasets/TXT_DB_v3'

ANNOTATIONS='datasets'
RE_ANN=$ANNOTATIONS/refer

if [ $TOKER = 'bert-large-cased' ]; then
    SUFFIX='large-cased'
elif [ $TOKER = 'bert-base-cased' ]; then
    SUFFIX='base-cased'
else
    echo "invalid tokenizer specified"
    # exit(1)
fi

# refcoco, refcoco+ 
for DATASET in 'refcoco' 'refcoco+'; do
    for SPLIT in 'train' 'val' 'testA' 'testB'; do
        python prepro.py --task re --bert $TOKER \
            --annotations $RE_ANN/${DATASET}/'refs(unc).p' \
                $RE_ANN/${DATASET}/instances.json \
                index/iid_to_ann_ids.json \
            --output $TXT_DB/${DATASET}_${SPLIT}_$SUFFIX.db
    done
done

# refcocog
DATASET='refcocog'
for SPLIT in 'train' 'val' 'test'; do
    python prepro.py --task re --bert $TOKER \
        --annotations $RE_ANN/${DATASET}/'refs(umd).p' \
            $RE_ANN/${DATASET}/instances.json \
            index/iid_to_ann_ids.json \
        --output $TXT_DB/${DATASET}_${SPLIT}_$SUFFIX.db
done


# DATASET='refcoco'
# SPLIT='train'
# python prepro.py --task re --bert $TOKER \
#     --annotations $RE_ANN/${DATASET}/'refs(unc).p' \
#         $RE_ANN/${DATASET}/instances.json \
#         index/iid_to_ann_ids.json \
#     --output $TXT_DB/${DATASET}_${SPLIT}_$SUFFIX.db