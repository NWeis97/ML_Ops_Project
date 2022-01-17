#!/bin/bash

torch-model-archiver \
    --model-name my_fancy_model \
    --version 1.0 \
    --serialized-file deployable_models/resnet18_pretrained.pt \
    --export-path model_store \
    --extra-files torchserve/index_to_name.json \
    --handler image_classifier