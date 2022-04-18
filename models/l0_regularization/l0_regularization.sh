MODEL_TYPE=masked_bert
MODEL_NAME_OR_PATH=bert-base-uncased
TASK_NAME=SST-2
NUM_TRAIN_EPOCHS=10
SAVE_STEPS=10000
PER_GPU_TRAIN_BATCH_SIZE=16
PER_GPU_EVAL_BATCH_SIZE=16
WARMUP_STEPS=5400
SEED=101
PRUNING_METHOD=l0

DATA_DIR=D:/Programming/mastersthesis/models/l0_regularization/datasets/glue_data/SST-2
CACHE_DIR=D:/Programming/mastersthesis/models/l0_regularization/cache_dir

FINAL_LAMBDA=0
OUTPUT_DIR=D:/Programming/mastersthesis/models/l0_regularization/results/bert_3/lambda0

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --save_steps $SAVE_STEPS \
    --do_eval --do_lower_case \
    --model_type masked_bert \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_gpu_train_batch_size 16 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --pruning_method $PRUNING_METHOD \
    --mask_scores_learning_rate 1e-2 \
    --initial_threshold 1 --final_threshold 1. \
    --mask_init constant --mask_scale 2.197 \
    --regularization l0 --final_lambda $FINAL_LAMBDA \
    --initial_warmup 1 --final_warmup 1 --warmup_steps $WARMUP_STEPS \
    --eval_all_checkpoints --seed $SEED

FINAL_LAMBDA=1
OUTPUT_DIR=D:/Programming/mastersthesis/models/l0_regularization/results/bert_3/lambda1

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --save_steps $SAVE_STEPS \
    --do_eval --do_lower_case \
    --model_type masked_bert \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_gpu_train_batch_size 16 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --pruning_method $PRUNING_METHOD \
    --mask_scores_learning_rate 1e-2 \
    --initial_threshold 1 --final_threshold 1. \
    --mask_init constant --mask_scale 2.197 \
    --regularization l0 --final_lambda $FINAL_LAMBDA \
    --initial_warmup 1 --final_warmup 1 --warmup_steps $WARMUP_STEPS \
    --eval_all_checkpoints --seed $SEED

FINAL_LAMBDA=5
OUTPUT_DIR=D:/Programming/mastersthesis/models/l0_regularization/results/bert_3/lambda5

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --save_steps $SAVE_STEPS \
    --do_eval --do_lower_case \
    --model_type masked_bert \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_gpu_train_batch_size 16 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --pruning_method $PRUNING_METHOD \
    --mask_scores_learning_rate 1e-2 \
    --initial_threshold 1 --final_threshold 1. \
    --mask_init constant --mask_scale 2.197 \
    --regularization l0 --final_lambda $FINAL_LAMBDA \
    --initial_warmup 1 --final_warmup 1 --warmup_steps $WARMUP_STEPS \
    --eval_all_checkpoints --seed $SEED

FINAL_LAMBDA=25
OUTPUT_DIR=D:/Programming/mastersthesis/models/l0_regularization/results/bert_3/lambda25

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --save_steps $SAVE_STEPS \
    --do_eval --do_lower_case \
    --model_type masked_bert \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_gpu_train_batch_size 16 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --pruning_method $PRUNING_METHOD \
    --mask_scores_learning_rate 1e-2 \
    --initial_threshold 1 --final_threshold 1. \
    --mask_init constant --mask_scale 2.197 \
    --regularization l0 --final_lambda $FINAL_LAMBDA \
    --initial_warmup 1 --final_warmup 1 --warmup_steps $WARMUP_STEPS \
    --eval_all_checkpoints --seed $SEED

FINAL_LAMBDA=50
OUTPUT_DIR=D:/Programming/mastersthesis/models/l0_regularization/results/bert_3/lambda50

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --save_steps $SAVE_STEPS \
    --do_eval --do_lower_case \
    --model_type masked_bert \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_gpu_train_batch_size 16 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --pruning_method $PRUNING_METHOD \
    --mask_scores_learning_rate 1e-2 \
    --initial_threshold 1 --final_threshold 1. \
    --mask_init constant --mask_scale 2.197 \
    --regularization l0 --final_lambda $FINAL_LAMBDA \
    --initial_warmup 1 --final_warmup 1 --warmup_steps $WARMUP_STEPS \
    --eval_all_checkpoints --seed $SEED

FINAL_LAMBDA=250
OUTPUT_DIR=D:/Programming/mastersthesis/models/l0_regularization/results/bert_3/lambda250

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --save_steps $SAVE_STEPS \
    --do_eval --do_lower_case \
    --model_type masked_bert \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_gpu_train_batch_size 16 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --pruning_method $PRUNING_METHOD \
    --mask_scores_learning_rate 1e-2 \
    --initial_threshold 1 --final_threshold 1. \
    --mask_init constant --mask_scale 2.197 \
    --regularization l0 --final_lambda $FINAL_LAMBDA \
    --initial_warmup 1 --final_warmup 1 --warmup_steps $WARMUP_STEPS \
    --eval_all_checkpoints --seed $SEED