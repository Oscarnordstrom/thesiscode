MODEL_TYPE=masked_bert
MODEL_NAME_OR_PATH=bert-base-multilingual-cased
TASK_NAME=SST-2
NUM_TRAIN_EPOCHS=10
SAVE_STEPS=10000
PER_GPU_TRAIN_BATCH_SIZE=16
PER_GPU_EVAL_BATCH_SIZE=16
WARMUP_STEPS=5400
PRUNING_METHOD=sigmoied_threshold

DATA_DIR=D:/Programming/mastersthesis/models/soft_movement_pruning/datasets/glue_data/SST-2
CACHE_DIR=D:/Programming/mastersthesis/models/soft_movement_pruning/cache_dir

FINAL_LAMBDA=0.8
OUTPUT_DIR=D:/Programming/mastersthesis/models/soft_movement_pruning/results/mbert/lambda08

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --save_steps $SAVE_STEPS \
    --do_train --do_eval --do_lower_case \
    --model_type masked_bert \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_gpu_train_batch_size 16 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --pruning_method $PRUNING_METHOD \
    --mask_scores_learning_rate 1e-2 \
    --initial_threshold 0 --final_threshold 0.1 \
    --mask_init constant --mask_scale 0 \
    --initial_warmup 1 --final_warmup 1 --warmup_steps $WARMUP_STEPS \
    --eval_all_checkpoints \
    --regularization l1 --final_lambda $FINAL_LAMBDA

FINAL_LAMBDA=4.5
OUTPUT_DIR=D:/Programming/mastersthesis/models/soft_movement_pruning/results/mbert/lambda4_5

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --save_steps $SAVE_STEPS \
    --do_train --do_eval --do_lower_case \
    --model_type masked_bert \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_gpu_train_batch_size 16 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --pruning_method $PRUNING_METHOD \
    --mask_scores_learning_rate 1e-2 \
    --initial_threshold 0 --final_threshold 0.1 \
    --mask_init constant --mask_scale 0 \
    --initial_warmup 1 --final_warmup 1 --warmup_steps $WARMUP_STEPS \
    --eval_all_checkpoints \
    --regularization l1 --final_lambda $FINAL_LAMBDA

FINAL_LAMBDA=3
OUTPUT_DIR=D:/Programming/mastersthesis/models/soft_movement_pruning/results/mbert/lambda3

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --save_steps $SAVE_STEPS \
    --do_train --do_eval --do_lower_case \
    --model_type masked_bert \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_gpu_train_batch_size 16 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --pruning_method $PRUNING_METHOD \
    --mask_scores_learning_rate 1e-2 \
    --initial_threshold 0 --final_threshold 0.1 \
    --mask_init constant --mask_scale 0 \
    --initial_warmup 1 --final_warmup 1 --warmup_steps $WARMUP_STEPS \
    --eval_all_checkpoints \
    --regularization l1 --final_lambda $FINAL_LAMBDA