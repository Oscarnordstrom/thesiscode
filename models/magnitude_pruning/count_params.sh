SERIALIZATION_DIR=D:/Programming/mastersthesis/models/magnitude_pruning/results/mbert/threshold003

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/counts_parameters.py \
    --threshold 0.03 \
    --pruning_method magnitude \
    --serialization_dir $SERIALIZATION_DIR