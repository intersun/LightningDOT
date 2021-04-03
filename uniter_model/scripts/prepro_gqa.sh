GQA_ANN='/db/raw_data/GQA/questions1.2/train_all_questions/'
SUFFIX="base-cased"
TXT_DB='/db/TXT_DB_v3'
SPLIT="train"

for i in 2 3 4 5 6 7 8 9; do
	python prepro.py --task gqa \
		--annotations $GQA_ANN/${SPLIT}_all_questions_$i.json \
		--output $TXT_DB/pretrain_gqa_${SPLIT}_${i}_$SUFFIX.db
done


