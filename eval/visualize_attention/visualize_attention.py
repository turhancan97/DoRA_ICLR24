# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
sys.path.append('../')
sys.path.append('../../')

import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

import utils
import vision_transformer_viz as vits

company_colors = [
    (0,160,215), # blue
    (220,55,60), # red
    (245,180,0), # yellow
    (10,120,190), # navy
    (40,150,100), # green
    (135,75,145), # purple
]
company_colors = [(float(c[0]) / 255.0, float(c[1]) / 255.0, float(c[2]) / 255.0) for c in company_colors]

mapping = {
    2642: (0, 1, 2),
    700: (4, 5),
    1837: (2, 4),
    1935: (0, 1, 4),
    2177: (0,1,2),
    3017: (0,1,2),
    3224: (1, 2, 5),
    3340: (2, 1, 3),
    3425: (0, 1, 2),
    1954: (0,1,2),
    2032: (4,0,5),
    3272: (0,1,2),
    3425: (0,1,2),
    3695: (1,5),
    1791:(1,2,5),
    385 : (1, 5),
    4002: (0,1,2,3)
}

def apply_mask2(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    t= 0.2
    mi = np.min(mask)
    ma = np.max(mask)
    mask = (mask - mi) / (ma - mi)
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * np.sqrt(mask) * (mask>t))+ alpha * np.sqrt(mask) * (mask>t) * color[c] * 255
    return image


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return

def show_attn_color(image, attentions, th_attn, index=None, head=[0,1,2,3,4,5]):
        M = image.max()
        m = image.min()
        span = 64
        image = ((image - m) / (M-m)) * span + (256 - span)
        image = image.mean(axis=2)
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        
        for j in head:
            m = attentions[j]
            m *= th_attn[j]
            attentions[j] = m
        mask = np.stack([attentions[j] for j in head])
        
        blur = False
        contour = False
        alpha = 1
        figsize = tuple([i / 100 for i in args.image_size])
        fig = plt.figure(figsize=figsize, frameon=False, dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax = plt.gca()

        if len(mask.shape) == 3:
            N = mask.shape[0]
        else:
            N = 1
            mask = mask[None, :, :]

        # AJ
        for i in range(N):
            mask[i] = mask[i] * ( mask[i] == np.amax(mask, axis=0))
        a = np.cumsum(mask, axis=0)
        for i in range(N):
            mask[i] = mask[i] * (mask[i] == a[i])
        
        # if imid == 3340:
        #     tmp = company_colors[2]
        #     company_colors[2] = company_colors[1]
        #     company_colors[1] = tmp
        colors = company_colors[:N]

        # Show area outside image boundaries.
        height, width = image.shape[:2]
        margin = 0
        ax.set_ylim(height + margin, -margin)
        ax.set_xlim(-margin, width + margin)
        ax.axis('off')
        masked_image = 0.1*image.astype(np.uint32).copy()
        for i in range(N):
            color = colors[i]
            _mask = mask[i]
            if blur:
                _mask = cv2.blur(_mask,(10,10))
            # Mask
            masked_image = apply_mask2(masked_image, _mask, color, alpha)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            if contour:
                padded_mask = np.zeros(
                    (_mask.shape[0] + 2, _mask.shape[1] + 2))#, dtype=np.uint8)
                padded_mask[1:-1, 1:-1] = _mask
                contours = find_contours(padded_mask, 0.5)
                for verts in contours:
                    # Subtract the padding and flip (y, x) to (x, y)
                    verts = np.fliplr(verts) - 1
                    p = Polygon(verts, facecolor="none", edgecolor=color)
                    ax.add_patch(p)
        ax.imshow(masked_image.astype(np.uint8), aspect='auto')
        ax.axis('image')
        #fname = os.path.join(output_dir, 'bnw-{:04d}'.format(imid))
        prefix = f'id{index}_' if index is not None else ''
        fname = os.path.join(args.output_dir, "attn_color.png")
        fig.savefig(fname)
        attn_color = Image.open(fname)

        return attn_color

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--checkpoint', default='../../model/venice/checkpoint_all_100.pth', type=str,
        help="Path to pretrained weights to load.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument("--image_path", default=None, type=str, help="Path of the image to load.")
    parser.add_argument("--image_size", default=(224*5, 224*5), type=int, nargs="+", help="Resize image.")
    parser.add_argument('--output_dir', default='.', help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    utils.load_pretrained_weights(model, args.checkpoint, args.checkpoint_key, args.arch, args.patch_size)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print("Please use the `--image_path` argument to indicate the path of the image you wish to visualize.")
        print("Since no image path have been provided, we take the first image in our paper.")
        response = requests.get("https://dl.fbaipublicfiles.com/dino/img.png")
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            img_original = img.copy()
            img_original = img_original.resize(args.image_size)

    else:
        print(f"Provided image path {args.image_path} is non valid.")
        sys.exit(1)
    transform = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform(img)

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // args.patch_size
    h_featmap = img.shape[-1] // args.patch_size

    attentions = model.get_last_selfattention(img.to(device))

    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    if args.threshold is not None:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - args.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

    # save attentions heatmaps
    os.makedirs(args.output_dir, exist_ok=True)
    torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
    fig, axes = plt.subplots(1, nh+1, figsize=(20, 5))
    ax = axes[0]
    ax.imshow(img_original)
    ax.set_title("Main Image")
    for j in range(1, nh+1):
        ax = axes[j]
        ax.imshow(attentions[j-1])
        ax.set_title(f"Head {j-1}")

    fig.suptitle('Attention Heads Visualization')
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    output_path = os.path.join(args.output_dir, "attn_heads.png")
    plt.savefig(output_path)

    show_attn_color(img.squeeze(0).permute(1, 2, 0).cpu().numpy(), attentions, th_attn, index=None)

    if args.threshold is not None:
        image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
        for j in range(nh):
            display_instances(image, th_attn[j], fname=os.path.join(args.output_dir,'masks/',"mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)