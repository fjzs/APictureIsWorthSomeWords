import pandas as pd
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from doc2img.dataloader import get_raw_dataset
from doc2img.generating_images import generate_image
from doc2img.summarization import get_summary
from doc2img.clip_inference import get_pretrained_clip_scores
import yaml
import matplotlib.pyplot as plt
import sys
from pdb import set_trace as bp

config_file = sys.argv[1]
with open(config_file) as cf_file:
    config = yaml.safe_load( cf_file.read())

DATASET_TYPE = config['datasets']['type']  # poems or nyt
print("Dataset : ", DATASET_TYPE)
dataset_path_mini = config['datasets'][DATASET_TYPE + '_mini']
dataset_path_full = config['datasets'][DATASET_TYPE + '_full']
df_mini = get_raw_dataset(DATASET_TYPE, dataset_path_mini, max_examples=None)
df_full = get_raw_dataset(DATASET_TYPE, dataset_path_full, max_examples=None)

print("Summarization method : ", config['summary_method'])

#generating summaries
print("Generating summaries")
df = get_summary(df_full, df_mini, config)

print("Generating Images")
df = generate_image(df, config['prompts'])

print("Getting CLIP scores")
df = get_pretrained_clip_scores(df, config)

print("Saving DF with summary, image paths and clip scores")
df.to_csv(f"{config['image_generation']['save_folder']}/df.csv", index=False)