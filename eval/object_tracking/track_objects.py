import sys
sys.path.append('../../')

import sys
import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from vision_transformer import DINOHead
import torchvision.transforms._transforms_video as transforms_video
import vision_transformer as vits
from einops import rearrange
from PIL import Image
import torch.backends.cudnn as cudnn
import utils
from WTDataloader import WT_dataset_1vid
import matplotlib.pyplot as plt

def get_args_parser():
    parser = argparse.ArgumentParser('DORA Inference', add_help=False)
    parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base'])
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--input_video', default='input/8.mp4', type=str, help='Path to the input video file.')
    parser.add_argument('--output_video', default='output/output.mp4', type=str, help='Path to save the output video.')
    parser.add_argument('--checkpoint', default='../../model/venice/checkpoint_all_100.pth', type=str, help='Path to the model checkpoint.')
    parser.add_argument('--out_dim', default=65536, type=int)
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag)
    parser.add_argument('--image_scale', default=3, type=int)
    return parser

def load_model(args):
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    embed_dim = model.embed_dim
    model = utils.MultiCropWrapper(model, DINOHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head), image_scale = args.image_scale)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    state_dict = state_dict["teacher"]
    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(args.checkpoint, msg))
    # utils.load_pretrained_weights(model, args.checkpoint, "teacher", args.arch, args.patch_size)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.cuda()
    return model

def sample_frames(video_path, frame_per_clip, step_between_clips):
    transform = DataAugmentationDINO()

    dataset = WT_dataset_1vid(video_path, 
                frame_per_clip,  
                step_between_clips,
                transform=transform)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
        drop_last=True
    )
    print(f"Data loaded")
    return data_loader

def visualize_attention(model, data_loader, patch_size):
    # move images to gpu
    images = [im.cuda(non_blocking=True) for im in data_loader]
    # print('Iteration: ',it)
    print('Length of the Images: ', len(images))
    print('Shape of a Image: ', images[0].shape)
    with torch.no_grad():
        _, teacher_patches, attn, query, key = model(images[:1], return_track=True)
        masked_img = utils.MOT(attn, query, key, teacher_patches, patch_size, images[:1])
    print(f"Attention shapes: attn={attn.shape}, query={query.shape}, key={key.shape}")
    print(f"Generated attention maps with shape: {masked_img.shape}")
    # create a figure to visualize the attention maps object's are rows and frames are columns. for only first batch
    fig, axs = plt.subplots(masked_img.shape[0], masked_img.shape[2], figsize=(20, 20))
    for obj_idx in range(masked_img.shape[0]):
        for frame_idx in range(masked_img.shape[2]):
            attn_map = masked_img[obj_idx, 0, frame_idx].mean(axis=0).cpu().numpy()
            axs[obj_idx, frame_idx].imshow(attn_map)
            axs[obj_idx, frame_idx].axis('off')
    fig.savefig("attention_maps.png")
    return masked_img

def save_video(frames, masked_img, output_path):
    # num_objects, batch_size, num_frames, channels, height, width = masked_img.shape
    frames = frames[0][0]
    frames = frames.permute(0,2,3,1)
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    
    for frame_idx, frame in enumerate(frames):
        combined_frame = frame.cpu().numpy().copy()
        combined_frame = (combined_frame - combined_frame.min()) / (combined_frame.max() - combined_frame.min())
        combined_frame = (combined_frame * 255).astype(np.uint8)

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2GRAY)
        combined_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        # Create an empty attention map for 3 channels
        attn_map_combined = np.zeros((height, width, 3), dtype=np.uint8)
        
        for obj_idx in range(masked_img.shape[0]): # masked_img.shape[0]
            attn_map = masked_img[obj_idx, 0, frame_idx].mean(axis=0).cpu().numpy()
            attn_map = cv2.resize(attn_map, (width, height))
            # attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
            attn_map = (attn_map * 255).astype(np.uint8)
            attn_map = cv2.GaussianBlur(attn_map, (35, 35), 0)  # Apply Gaussian Blur
            attn_map_combined[:, :, obj_idx] = attn_map  # Assign to the respective channel

        combined_frame = cv2.addWeighted(combined_frame, 0.5, attn_map_combined, 0.5, 0)
        
        out.write(combined_frame)
    
    out.release()
    print(f"Saved visualization to {output_path}")

class DataAugmentationDINO(object):
    def __init__(self):
        self.init_aug = transforms.Compose([
            transforms_video.ToTensorVideo(),
            # transforms_video.RandomResizedCropVideo(300, (0.2, 1)),
            ])

        self.vid_crop = transforms.Compose([
            transforms.Resize((224*args.image_scale,224*args.image_scale))
            # transforms_video.RandomResizedCropVideo(224, (0.4, 1)),
            # transforms_video.RandomHorizontalFlipVideo(p=0.5)
            ])



        normalize = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.ToPILImage(),
            normalize,
        ])
    def __call__(self, image):
        crops = []

        # an aggresive cropping before DINO crops
        frames = self.init_aug(image)
        

        gc_frame1 = self.vid_crop(frames).permute(1,0,2,3)
        # gc_frame2 = self.vid_crop(frames).permute(1,0,2,3)

        num_frame = gc_frame1.shape[0]
        
        
        global_1 = torch.zeros(gc_frame1.shape)
        # global_2 = torch.zeros(gc_frame2.shape)
        
        for cnt in range(num_frame):
            global_1[cnt,:]= self.global_transfo1(gc_frame1[cnt,:])
            
            # global_2[cnt,:]= self.global_transfo2(gc_frame2[cnt,:])
        
        crops.append(global_1)
        # crops.append(global_2)
        return crops

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DORA Inference', parents=[get_args_parser()])
    args = parser.parse_args()
    cap = cv2.VideoCapture(args.input_video)
    step_between_clips = 10
    frame_per_clip = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / step_between_clips)
    model = load_model(args)
    frames = sample_frames(args.input_video, frame_per_clip, step_between_clips)
    frames = next(iter(frames))
    masked_img = visualize_attention(model, frames, args.patch_size)
    save_video(frames, masked_img, args.output_video)