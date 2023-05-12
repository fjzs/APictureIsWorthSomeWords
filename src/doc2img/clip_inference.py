import gc
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from transformers import DistilBertTokenizer, CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import pandas as pd

import config as CFG
from train import build_loaders
# from encoders.clip import CLIPModel
import albumentations as A
from PIL import Image


def get_pretrained_clip_scores(
        image_path,text
    ):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    image = Image.open(image_path)

    inputs = processor(text=[text], images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    # probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

    return logits_per_image


def get_image_embeddings(valid_df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)


    return model, torch.cat(valid_image_embeddings)

def get_image_query_similarity(image_path, query, model_path):
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()

    # Configure Image
    image = cv2.imread(f"{image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = A.Compose(
        [
            A.Resize(CFG.size, CFG.size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ]
    )
    image = transform(image=image)['image']
    image = torch.tensor(image).permute(2, 0, 1).float()
    image = image.unsqueeze(0)

    with torch.no_grad():
        image_features = model.image_encoder(image.to(CFG.device))
        image_embeddings = model.image_projection(image_features)

    # Configure Query
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    return dot_similarity[0][0]


def find_matches(model, image_embeddings, query, image_filenames, n=9):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    _, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]
    
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{CFG.image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")
    
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', type=str, default="dogs on the grass")
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--operation', type=str, choices=['search', 'similarity'], default='search')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model_path', type=str, default='best.pt')
    parser.add_argument('--n', type=int, default=3)
    parser.add_argument('--pretrained', type=bool, default=True)
    args = parser.parse_args()


    if args.operation == 'search':
        if args.data_path == None:
            raise ValueError("Data Path needs to be specified. Please set the data_path argument.")
        test_data = pd.read_csv(f"{args.data_path}/captions.txt")
        model, embeddings = get_image_embeddings(test_data, args.model_path)

        find_matches(
            model, 
            embeddings, 
            query=args.query, 
            image_filenames=test_data['image'].values, 
            n=args.n
        )

    else:
        if args.image_path == None:
            raise ValueError("Image Path needs to be specified. Please set the image_path argument.")
        
        if args.pretrained:
            similarity = get_pretrained_clip_scores(
                args.image_path,
                args.query
            )
        else:
            similarity = get_image_query_similarity(
                args.image_path, 
                args.query, 
                args.model_path
            )

        print("Similarity between Query and Image: {}".format(similarity))