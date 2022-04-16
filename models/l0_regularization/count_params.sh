SERIALIZATION_DIR=D:/Programming/mastersthesis/models/l0_regularization/results/mbert/lambda250

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/counts_parameters.py \
    --pruning_method l0 \
    --serialization_dir $SERIALIZATION_DIR
