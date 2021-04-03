function run_exp {
  GPU_ID=$1
  LR=$2
  BS=$3
  NUM_HARD=$4
  SAMPLEING_METHOD=$5
  OUTPUT_NAME=$6
  IMG_CKPT=$7
  TXT_CKPT=$8
  BIENCODER_CKPT=$9
  CAPTION_SCORE_WEIGHT=${10}
  PROJECT_NAME=${11}
  PROJECT_DIM=${12}
  RETRIEVAL_MODE=${13}
  SURFIX=${14}


  output_dir=$OUTPUT_NAME

  current_output_dir="${output_dir}/${LR}_${BS}_${NUM_HARD}_${SAMPLEING_METHOD}_${CAPTION_SCORE_WEIGHT}_${PROJECT_DIM}_${RETRIEVAL_MODE}_run${SURFIX}"
  mkdir -p $current_output_dir
  cmd="CUDA_VISIBLE_DEVICES=${GPU_ID} python train_itm.py --config ./config/flickr30k_ft_config_bert.json \
  --learning_rate $LR --train_batch_size $BS --output_dir $current_output_dir --fp16 \
  --num_hard_negatives $NUM_HARD --project_name $PROJECT_NAME --caption_score_weight $CAPTION_SCORE_WEIGHT  \
  --hard_negatives_sampling $SAMPLEING_METHOD --fp16_opt_level O2 --project_dim $PROJECT_DIM  \
  --img_checkpoint $IMG_CKPT --retrieval_mode $RETRIEVAL_MODE"

  if [ $BIENCODER_CKPT != "none" ];
  then
    cmd="$cmd --biencoder_checkpoint $BIENCODER_CKPT"
  fi
  cmd="$cmd > $current_output_dir/log 2>&1 &"
  echo "$cmd"
  eval "$cmd"
}


echo "**********************************************************************"
IMG_CKPT=/pretrain/alltask_ot_alldata.pt
TXT_CKPT=none
BIMODEL_CKPT=/pretrain/LightningDot.pt
LR=2e-5
BSZ=96
PROJECT_NAME=itm_flickr
OUTPUT_NAME=/storage/flickr-bert-two_stream
SAMPLEING_METHOD=none
PROJECT_DIM=768
RETRIEVAL_MODE=both
#         GPU_ID    LR      BS       NUM_HARD     SAMPLING_METHOD        OUTPUT_NAME       IMG_CKPT   TXT_CKPT   BIENCODER_CKPT     CAPTION_SCORE_WEIGHT     PROJECT_NAME       PROJECT_DIM      RETRIEVA_MODE         SURFIX
run_exp     0       1e-5   $BSZ        0          $SAMPLEING_METHOD      $OUTPUT_NAME      $IMG_CKPT  $TXT_CKPT  $BIMODEL_CKPT             0.0               $PROJECT_NAME      $PROJECT_DIM     $RETRIEVAL_MODE         1
run_exp     1       2e-5   $BSZ        0          $SAMPLEING_METHOD      $OUTPUT_NAME      $IMG_CKPT  $TXT_CKPT  $BIMODEL_CKPT             0.0               $PROJECT_NAME      $PROJECT_DIM     $RETRIEVAL_MODE         1
run_exp     2       5e-5   $BSZ        0          $SAMPLEING_METHOD      $OUTPUT_NAME      $IMG_CKPT  $TXT_CKPT  $BIMODEL_CKPT             0.0               $PROJECT_NAME      $PROJECT_DIM     $RETRIEVAL_MODE         1
run_exp     3       8e-5   $BSZ        0          $SAMPLEING_METHOD      $OUTPUT_NAME      $IMG_CKPT  $TXT_CKPT  $BIMODEL_CKPT             0.0               $PROJECT_NAME      $PROJECT_DIM     $RETRIEVAL_MODE         1
