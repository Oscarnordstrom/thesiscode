SERIALIZATION_DIR=D:/Programming/mastersthesis/movement_pruning/serialization_dir
DATA_DIR=D:/Programming/mastersthesis/movement_pruning/datasets/sst
MODEL_TYPE=bert
MODEL_NAME_OR_PATH=bert-base-uncased
TASK_NAME=sst-2
OUTPUT_DIR=D:/Programming/mastersthesis/movement_pruning/results_SST
CACHE_DIR=D:/Programming/mastersthesis/movement_pruning/cache_dir
PER_GPU_TRAIN_BATCH_SIZE=16
PER_GPU_EVAL_BATCH_SIZE=16
LEARNING_RATE=3e-5
MASK_SCORES_LEARNING_RATE=1e-2
INITIAL_THRESHOLD=1
FINAL_THRESHOLD=0.5
INITIAL_WARMUP=1
FINAL_WARMUP=2
PRUNING_METHOD=topK
MASK_INIT=constant
MASK_SCALE=0.
SAVE_STEPS=8000
NUM_TRAIN_EPOCHS=3
WARMUP_STEPS=5400
SEED=30


python D:/Programming/mastersthesis/transformers/examples/research_projects/movement-pruning/masked_run_glue.py \
    --data_dir $DATA_DIR \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --task_name $TASK_NAME \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --do_train \
    --do_eval \
    --do_lower_case \
    --per_gpu_train_batch_size $PER_GPU_TRAIN_BATCH_SIZE \
    --per_gpu_eval_batch_size $PER_GPU_EVAL_BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --mask_scores_learning_rate $MASK_SCORES_LEARNING_RATE \
    --initial_threshold $INITIAL_THRESHOLD \
    --final_threshold $FINAL_THRESHOLD \
    --initial_warmup $INITIAL_WARMUP \
    --final_warmup $FINAL_WARMUP \
    --pruning_method $PRUNING_METHOD \
    --mask_init constant \
    --mask_scale 0. \
    --mask_init $MASK_INIT \
    --mask_scale $MASK_SCALE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --warmup_steps $WARMUP_STEPS \
    --overwrite_output_dir \
    --overwrite_cache \
    --seed $SEED \
    --save_steps $SAVE_STEPS