from pathlib import Path
import argparse
import random

import torch
import torch.autograd as autograd
import torch.nn.functional as F

from torch.nn import MSELoss
import torchvision.transforms.v2 as transforms

import numpy as np
import cv2

from tqdm import tqdm

from backbones.iresnet import iresnet100, iresnet50, iresnet18
from backbones.bn import BN_Model


def get_model(
    nn_architecture : str,
    rank,
    nn_weights_path : str,
    embedding_size : int = 512
):
    
    if nn_architecture == "iresnet100":
        backbone = iresnet100(num_features=embedding_size, use_se=False).to(rank)
    elif nn_architecture == "iresnet50":
        backbone = iresnet50(num_features=embedding_size, dropout=0.4, use_se=False).to(rank)
    else:
        raise ValueError("Unknown model architecture given.")
    
    backbone.load_state_dict(torch.load(nn_weights_path, map_location=torch.device(rank)))
    backbone.return_intermediate = True
    backbone.eval()
    
    backbone = BN_Model(backbone, rank)

    return backbone


def write_score(output_file_path, # File to write scores to
                quality_scores, # GraFIQs quality scores
                image_paths, # List of image paths (either Path or str)
                replace_str  = None, # String to be replaced in image path, e.g. replace_str="/replace/"
                replace_with = None # String that will replace previous id, e.g. replace_with="/newpath/" -> /replace/img01.jpg -> /newpath/img01.jpg
                ):
    
    with open(output_file_path, "w") as f:
        for idx in range(len(quality_scores)):
            image_path = str(image_paths[idx])
            
            if replace_str is not None and replace_with is not None:
                image_path = image_path.replace(replace_str, replace_with)
        
            f.write(f"{image_path} {quality_scores[idx]}\n")


def main(args):
    print(args)
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    rank = torch.device(f"cuda:{args.gpu}")
    
    images = sorted(Path(args.image_path).glob(f"*.{args.image_extension}"))
    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    model_backbone = get_model(
                                nn_architecture=args.backbone,
                                rank=rank,
                                nn_weights_path=args.weights
                              )

    image_transforms = transforms.Compose(
                            [
                            transforms.ToImage(),
                            transforms.Resize(size=(112,112),
                                              interpolation=transforms.InterpolationMode.BILINEAR,
                                              antialias=True),
                            transforms.ToDtype(torch.float32, scale=True),
                            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])   
                            ]
                        )

    scores = {k:[] for k in ["image", "block1", "block2", "block3", "block4"]}
    
    for path in tqdm(images):
        image = cv2.imread(str(path))
        if args.bgr2rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image_transforms(image).unsqueeze(0).to(rank).requires_grad_(True)
        
        bn_score, (emb, block1, block2, block3, block4, bn) = model_backbone.get_BN(image)
        grads = autograd.grad(
                    outputs=bn_score,
                    inputs=[image, block1, block2, block3, block4]
                )
        
        for idx, key in enumerate(["image", "block1", "block2", "block3", "block4"]):
            grad_tensor = grads[idx][0].cpu()
            scores[key].append( float( torch.abs(grad_tensor).sum() ) )

    for key in ["image", "block1", "block2", "block3", "block4"]:
        write_score(
                    output_folder / f"GraFIQs_{key}.txt",
                    scores[key],
                    images,
                    args.path_replace,
                    args.path_replace_with
        )
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='GraFIQs')
    
    parser.add_argument('--image-path', type=str, help='Path to images.')
    parser.add_argument('--image-extension', type=str, default="jpg", help='Extension/File type of images (e.g. jpg, png).')
    parser.add_argument('--output-dir', type=str, help='Directory to write score files to (will be created if it does not exist).')
    parser.add_argument('--backbone', type=str, choices=["iresnet50", "iresnet100"], help='Backbone architecture to use.')
    parser.add_argument('--weights', type=str, help='Path to backbone architecture weights.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use.')
    parser.add_argument('--path-replace', type=str, default=None, help='Prefix of image path which shall be replaced.')
    parser.add_argument('--path-replace-with', type=str, default=None, help='String that replaces prefix given in --path-replace.')
    parser.add_argument('--bgr2rgb', action='store_true', help='If specified, changes color space of CV2 image from BGR to RGB.')

    main(parser.parse_args())
    