import time
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
import torch
import yaml
import os

#setting device to gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#reading config file
config_file = './config.yaml'
with open(config_file) as cf_file:
    config = yaml.safe_load( cf_file.read())

#model and seed
model_id = config['image_generation']['model_id']
seed = config['image_generation']['seed']
inference_steps = config['image_generation']['inference_steps']

#save parameters
save_folder = config['image_generation']['save_folder']
save_flag = config['image_generation']['save_flag']

#creating model
pipe = StableDiffusionPipeline.from_pretrained(model_id)



def generate_image(df):
    
    #Inputs:
    #  df: dataframe contains summaries
    #Outputs:
    #  df: datarframe containing summaries and associated saved image paths
    
    #Creating new column
    df['img_path'] = None
    
    os.mkdir(save_folder)
    
    #Generating images one by one
    for index,prompt in enumerate(df['summary']):
        
        #setting seed
        generator = torch.Generator(device).manual_seed(seed)
        
        image = pipe(prompt, generator=generator, num_inference_steps=inference_steps, output_type="np").images
        
        #saving images 
        if save_flag:
            save_path = os.path.join(save_folder,str(index) +'.jpg')
            df['img_path'][index] = save_path
            plt.imsave(save_path, image[0])
     
    return df