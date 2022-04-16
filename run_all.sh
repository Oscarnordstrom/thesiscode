MODEL_TYPE=masked_bert
MODEL_NAME_OR_PATH=bert-base-uncased
TASK_NAME=sst-2
NUM_TRAIN_EPOCHS=10
SAVE_STEPS=2000
PER_GPU_TRAIN_BATCH_SIZE=16
PER_GPU_EVAL_BATCH_SIZE=16
FINAL_THRESHOLD=0.03

DATA_DIR=D:/Programming/mastersthesis/models/magnitude_pruning/datasets/sst
OUTPUT_DIR=D:/Programming/mastersthesis/models/magnitude_pruning/results
CACHE_DIR=D:/Programming/mastersthesis/models/magnitude_pruning/cache_dir
PRUNING_METHOD=magnitude


python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --do_train \
    --do_eval \
    --do_lower_case \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --pruning_method $PRUNING_METHOD \
    --per_gpu_train_batch_size $PER_GPU_TRAIN_BATCH_SIZE \
    --per_gpu_eval_batch_size $PER_GPU_EVAL_BATCH_SIZE \
    --eval_all_checkpoints \
    --save_steps $SAVE_STEPS \
    --final_threshold $FINAL_THRESHOLD \
    --initial_warmup 1 --final_warmup 2 --warmup_steps 5400

DATA_DIR=D:/Programming/mastersthesis/models/l0_regularization/datasets/sst
OUTPUT_DIR=D:/Programming/mastersthesis/models/l0_regularization/results
CACHE_DIR=D:/Programming/mastersthesis/models/l0_regularization/cache_dir
PRUNING_METHOD=l0

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --do_train \
    --do_eval \
    --do_lower_case \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --pruning_method $PRUNING_METHOD \
    --per_gpu_train_batch_size $PER_GPU_TRAIN_BATCH_SIZE \
    --per_gpu_eval_batch_size $PER_GPU_EVAL_BATCH_SIZE \
    --eval_all_checkpoints \
    --save_steps $SAVE_STEPS \
    --final_threshold $FINAL_THRESHOLD \
    --initial_warmup 1 --final_warmup 2 --warmup_steps 5400

DATA_DIR=D:/Programming/mastersthesis/models/movement_pruning/datasets/sst
OUTPUT_DIR=D:/Programming/mastersthesis/models/movement_pruning/results
CACHE_DIR=D:/Programming/mastersthesis/models/movement_pruning/cache_dir
PRUNING_METHOD=topK

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --do_train \
    --do_eval \
    --do_lower_case \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --pruning_method $PRUNING_METHOD \
    --per_gpu_train_batch_size $PER_GPU_TRAIN_BATCH_SIZE \
    --per_gpu_eval_batch_size $PER_GPU_EVAL_BATCH_SIZE \
    --eval_all_checkpoints \
    --save_steps $SAVE_STEPS \
    --final_threshold $FINAL_THRESHOLD \
    --initial_warmup 1 --final_warmup 2 --warmup_steps 5400

DATA_DIR=D:/Programming/mastersthesis/models/soft_movement_pruning/datasets/sst
OUTPUT_DIR=D:/Programming/mastersthesis/models/soft_movement_pruning/results
CACHE_DIR=D:/Programming/mastersthesis/models/soft_movement_pruning/cache_dir
PRUNING_METHOD=sigmoied_threshold

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --do_train \
    --do_eval \
    --do_lower_case \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --pruning_method $PRUNING_METHOD \
    --per_gpu_train_batch_size $PER_GPU_TRAIN_BATCH_SIZE \
    --per_gpu_eval_batch_size $PER_GPU_EVAL_BATCH_SIZE \
    --eval_all_checkpoints \
    --save_steps $SAVE_STEPS \
    --final_threshold $FINAL_THRESHOLD \
    --initial_warmup 1 --final_warmup 2 --warmup_steps 5400