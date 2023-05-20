#!/bin/bash

config_dir="doc2img/configs"
for config in "$config_dir"/*
do
    if [[ "$config" == *"hf"* ]]; then
        continue
    fi
    echo "$config"
    cat $config > config.yaml
    python main.py $config
done