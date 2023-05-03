import time
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
import torch
import yaml

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

#creating model
pipe = StableDiffusionPipeline.from_pretrained(model_id)

#setting seed
generator = torch.Generator(device).manual_seed(seed)

def generate_image(prompt):
    image = pipe(prompt, generator=generator, num_inference_steps=inference_steps, output_type="np").images
    return image