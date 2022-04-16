SERIALIZATION_DIR=D:/Programming/mastersthesis/models/l0_regularization/results/testing

python D:/Programming/mastersthesis/models/transformers/examples/research_projects/movement-pruning/bertarize.py \
    --pruning_method l0 \
    --model_name_or_path $SERIALIZATION_DIR