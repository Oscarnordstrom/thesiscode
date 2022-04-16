SERIALIZATION_DIR=D:/Programming/mastersthesis/models/movement_pruning/results

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/bertarize.py \
    --pruning_method l0 \
    --threshold 0.1 \
    --model_name_or_path $SERIALIZATION_DIR