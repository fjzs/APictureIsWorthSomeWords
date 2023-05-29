# !/bin/bash

config_dir="doc2img/new_configs"
for config in "$config_dir"/*
do
    if [[ $config == *"hf_bart"* ]] || [[ $config == *"hf_pegasus"* ]]; then
        echo "Skipping $config"
        continue
    fi
    echo "$config"
    cat $config > config.yaml
    python main.py config.yaml
done

