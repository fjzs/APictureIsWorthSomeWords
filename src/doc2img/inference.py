from diffusers import StableDiffusionPipeline
import torch
from pdb import set_trace as bp
import argparse

def main(args):
    model_id = args.model_id
    pipe = StableDiffusionPipeline.from_pretrained(model_id, num_inference_steps=args.inf_steps, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    prompt = "retro serie of different cars with different colors and shapes, mdjrny-v4 style"
    image = pipe(prompt).images[0]
    image.save(args.img_save_path)


if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id", type=str, 
        default="CompVis/stable-diffusion-v1-4",
        dest="model_id",
        help="Model used to generate image"
    )
    parser.add_argument(
        "--img-save-path", type=str,
        default="img0.jpg",
        dest="img_save_path",
        help="Path to save model"
    )
    parser.add_argument(
        "--inf-steps", type=int,
        default=30, dest="inf_steps",
        help="Number of inference steps"
    )
    # parser.add_argument(
    #     "--prompt-file", type=str,
    #     default=""
    # )
    args = parser.parse_args()
    main(args)