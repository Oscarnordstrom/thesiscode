SERIALIZATION_DIR=D:/Programming/mastersthesis/models/soft_movement_pruning/results/mbert_1/lambda08

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/counts_parameters.py \
    --pruning_method sigmoied_threshold \
    --threshold 0.1 \
    --serialization_dir $SERIALIZATION_DIR