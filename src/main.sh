#!/bin/bash

config_dir="doc2img/configs"
for config in "$config_dir"/*
do
    echo "$config"
    python main.py $config
done