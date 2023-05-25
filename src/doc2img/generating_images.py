import time
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
import torch
import yaml
import os
import pandas as pd

#setting device to gpu if available
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#reading config file
config_file = './config.yaml'
with open(config_file) as cf_file:
    config = yaml.safe_load(cf_file.read())

#model and seed
model_id = config['image_generation']['model_id']
seed = config['image_generation']['seed']
inference_steps = config['image_generation']['inference_steps']

#save parameters
save_folder = config['image_generation']['save_folder']
save_flag = config['image_generation']['save_flag']

#creating model
pipe = StableDiffusionPipeline.from_pretrained(model_id)
pipe = pipe.to(config['device'])



def generate_image(df, prompts):
    
    #Inputs:
    #  df: dataframe contains summaries
    #  prompts: list of required prompts
    #Outputs:
    #  df: datarframe containing summaries and associated saved image paths
    
    
    os.mkdir(save_folder)
    save_paths = {
        "prompt_"+str(idx): [] for idx in range(len(prompts))
    }
    
    #Generating images one by one
    for index,text in enumerate(df['summary']):

        for prompt_idx, prompt in enumerate(prompts):

            text = prompt + text
        
            #setting seed
            generator = torch.Generator(config['device']).manual_seed(seed)
            
            image = pipe(text, generator=generator, num_inference_steps=inference_steps, output_type="np").images
            
            #saving images 
            if save_flag:
                final_index = config['datasets']['type'] + "_" + str(index) + "_prompt_" + str(prompt_idx)
                save_path = os.path.join(save_folder,str(final_index) +'.jpg')
                save_paths["prompt_"+str(prompt_idx)].append(save_path)
                plt.imsave(save_path, image[0])

    df = pd.concat([df, pd.DataFrame.from_dict(save_paths)], axis=1)
    return df