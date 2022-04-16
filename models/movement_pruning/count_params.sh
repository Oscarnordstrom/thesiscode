SERIALIZATION_DIR=D:/Programming/mastersthesis/models/movement_pruning/results/mbert/threshold05

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/counts_parameters.py \
    --pruning_method topK \
    --threshold 0.5 \
    --serialization_dir $SERIALIZATION_DIR