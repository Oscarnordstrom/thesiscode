SERIALIZATION_DIR=D:/Programming/mastersthesis/models/magnitude_pruning/results/bert/threshold003

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/bertarize.py \
    --pruning_method magnitude \
    --threshold 0.03 \
    --model_name_or_path $SERIALIZATION_DIR