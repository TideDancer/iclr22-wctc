export LR=1e-4
export RATIO=$1
export BATCH_SIZE=16
export BATCH_ACC=2
export EPOCH=50
export WCTC=True
wandb off

python run_asr.py \
--output_dir ./output/timit.asr.$WCTC.$RATIO \
--overwrite_output_dir \
--wctc $WCTC \
--mask_ratio $RATIO \
--num_train_epochs $EPOCH \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--gradient_accumulation_steps $BATCH_ACC \
--evaluation_strategy="steps" \
--warmup_ratio 0.1 \
--save_steps="500" \
--eval_steps="500" \
--logging_steps="100" \
--learning_rate $LR \
--model_name_or_path="facebook/wav2vec2-base" \
--save_total_limit 1 \
--fp16 \
--dataset_name="timit_asr" \
--train_split_name="train" \
--validation_split_name="test" \
--orthography="timit" \
--preprocessing_num_workers=20 \
--verbose_logging \
--group_by_length 
