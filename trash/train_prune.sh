SERIALIZATION_DIR=D:/Programming/mastersthesis/movement_pruning/serialization_dir
DATA_DIR=D:/Programming/mastersthesis/movement_pruning/datasets/squad

python D:/Programming/mastersthesis/transformers/examples/research_projects/movement-pruning/masked_run_squad.py \
    --output_dir $SERIALIZATION_DIR \
    --data_dir $DATA_DIR \
    --train_file train-v1.1.json \
    --predict_file dev-v1.1.json \
    --do_train --do_eval --do_lower_case \
    --model_type masked_bert \
    --model_name_or_path bert-base-uncased \
    --per_gpu_train_batch_size 8 \
    --warmup_steps 5400 \
    --num_train_epochs 1 \
    --learning_rate 3e-5 --mask_scores_learning_rate 1e-2 \
    --initial_threshold 1 --final_threshold 0.15 \
    --initial_warmup 1 --final_warmup 2 \
    --overwrite_output_dir \
    --pruning_method topK --mask_init constant --mask_scale 0.