MODEL_TYPE=masked_bert
MODEL_NAME_OR_PATH=bert-base-uncased
TASK_NAME=SST-2
NUM_TRAIN_EPOCHS=10
SAVE_STEPS=10000
PER_GPU_TRAIN_BATCH_SIZE=16
PER_GPU_EVAL_BATCH_SIZE=16
WARMUP_STEPS=5400
PRUNING_METHOD=magnitude
SEED=101

DATA_DIR=D:/Programming/mastersthesis/models/soft_movement_pruning/datasets/glue_data/SST-2
CACHE_DIR=D:/Programming/mastersthesis/models/magnitude_pruning/cache_dir

OUTPUT_DIR=D:/Programming/mastersthesis/models/magnitude_pruning/results/bert/threshold003
FINAL_THRESHOLD=0.03

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --save_steps $SAVE_STEPS \
    --do_train --do_eval --do_lower_case \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_gpu_train_batch_size 16 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --pruning_method $PRUNING_METHOD \
    --mask_scores_learning_rate 1e-2 \
    --initial_threshold 1 --final_threshold $FINAL_THRESHOLD \
    --mask_init constant --mask_scale 2.197 \
    --initial_warmup 1 --final_warmup 1 --warmup_steps $WARMUP_STEPS \
    --eval_all_checkpoints --seed $SEED

OUTPUT_DIR=D:/Programming/mastersthesis/models/magnitude_pruning/results/bert/threshold015
FINAL_THRESHOLD=0.15

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --save_steps $SAVE_STEPS \
    --do_train --do_eval --do_lower_case \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_gpu_train_batch_size 16 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --pruning_method $PRUNING_METHOD \
    --mask_scores_learning_rate 1e-2 \
    --initial_threshold 1 --final_threshold $FINAL_THRESHOLD \
    --mask_init constant --mask_scale 2.197 \
    --initial_warmup 1 --final_warmup 1 --warmup_steps $WARMUP_STEPS \
    --eval_all_checkpoints --seed $SEED

OUTPUT_DIR=D:/Programming/mastersthesis/models/magnitude_pruning/results/bert/threshold025
FINAL_THRESHOLD=0.25

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --save_steps $SAVE_STEPS \
    --do_train --do_eval --do_lower_case \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_gpu_train_batch_size 16 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --pruning_method $PRUNING_METHOD \
    --mask_scores_learning_rate 1e-2 \
    --initial_threshold 1 --final_threshold $FINAL_THRESHOLD \
    --mask_init constant --mask_scale 2.197 \
    --initial_warmup 1 --final_warmup 1 --warmup_steps $WARMUP_STEPS \
    --eval_all_checkpoints --seed $SEED

OUTPUT_DIR=D:/Programming/mastersthesis/models/magnitude_pruning/results/bert/threshold05
FINAL_THRESHOLD=0.5

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --save_steps $SAVE_STEPS \
    --do_train --do_eval --do_lower_case \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_gpu_train_batch_size 16 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --pruning_method $PRUNING_METHOD \
    --mask_scores_learning_rate 1e-2 \
    --initial_threshold 1 --final_threshold $FINAL_THRESHOLD \
    --mask_init constant --mask_scale 2.197 \
    --initial_warmup 1 --final_warmup 1 --warmup_steps $WARMUP_STEPS \
    --eval_all_checkpoints --seed $SEED

OUTPUT_DIR=D:/Programming/mastersthesis/models/magnitude_pruning/results/bert/threshold07
FINAL_THRESHOLD=0.7

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --save_steps $SAVE_STEPS \
    --do_train --do_eval --do_lower_case \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_gpu_train_batch_size 16 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --pruning_method $PRUNING_METHOD \
    --mask_scores_learning_rate 1e-2 \
    --initial_threshold 1 --final_threshold $FINAL_THRESHOLD \
    --mask_init constant --mask_scale 2.197 \
    --initial_warmup 1 --final_warmup 1 --warmup_steps $WARMUP_STEPS \
    --eval_all_checkpoints --seed $SEED

OUTPUT_DIR=D:/Programming/mastersthesis/models/magnitude_pruning/results/bert/threshold09
FINAL_THRESHOLD=0.9

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --save_steps $SAVE_STEPS \
    --do_train --do_eval --do_lower_case \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_gpu_train_batch_size 16 \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --pruning_method $PRUNING_METHOD \
    --mask_scores_learning_rate 1e-2 \
    --initial_threshold 1 --final_threshold $FINAL_THRESHOLD \
    --mask_init constant --mask_scale 2.197 \
    --initial_warmup 1 --final_warmup 1 --warmup_steps $WARMUP_STEPS \
    --eval_all_checkpoints --seed $SEED