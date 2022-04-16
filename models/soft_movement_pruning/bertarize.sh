SERIALIZATION_DIR=D:/Programming/mastersthesis/models/soft_movement_pruning/results/testing

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/bertarize.py \
    --pruning_method sigmoied_threshold \
    --threshold 500 \
    --model_name_or_path $SERIALIZATION_DIR