import sys
sys.path.append('../')
sys.path.append('../../')

import argparse
import os
import cv2
import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
from vision_transformer import DINOHead
import torchvision.transforms._transforms_video as transforms_video
import vision_transformer_viz as vits
import utils
from WTDataloader import WT_dataset_1vid
import matplotlib.pyplot as plt

def get_args_parser():
    parser = argparse.ArgumentParser('DORA Inference Visualization', add_help=False)
    parser.add_argument('--arch', default='vit_small', type=str, choices=['vit_tiny', 'vit_small', 'vit_base'])
    parser.add_argument('--patch_size', default=16, type=int)
    parser.add_argument('--input_video', default='input/8.mp4', type=str, help='Path to the input video file.')
    parser.add_argument('--output_video', default='output/output.mp4', type=str, help='Path to save the output video.')
    parser.add_argument('--checkpoint', default='../../model/venice/checkpoint_venice_300.pth', type=str, help='Path to the model checkpoint.')
    parser.add_argument('--out_dim', default=65536, type=int)
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag)
    parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    return parser

def load_model(args):
    """
    Loads a pre-trained model for object tracking.
    """
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    # embed_dim = model.embed_dim
    # model = utils.MultiCropWrapper(model, DINOHead(embed_dim, args.out_dim, use_bn=args.use_bn_in_head))

    utils.load_pretrained_weights(model, args.checkpoint, "teacher", args.arch, args.patch_size)
    for p in model.parameters():
        p.requires_grad = False
    model.cuda()
    model.eval()
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
    print("Data loaded")
    return data_loader

def visualize_attention(model, data_loader, patch_size, viz_atn=False):
    """
    Visualizes the attention maps of a given model on a set of images.

    Args:
        model (nn.Module): The model for which attention maps are to be visualized.
        data_loader (DataLoader): The data loader containing the images.
        patch_size (int): The size of the patches used by the model.
        viz_atn (bool, optional): Whether to visualize the attention maps using matplotlib. 
                                  Defaults to False.

    Returns:
        torch.Tensor: A tensor containing the attention maps for each image in the data loader.
                      Shape: [num_images, num_heads, width, height]
    """
    # move images to gpu
    images = [im.cuda(non_blocking=True) for im in data_loader]
    # print('Iteration: ',it)
    print('Length of the Images: ', len(images))
    print('Shape of a Image: ', images[0].shape)
    attention_list = []
    for i, img in enumerate(images[0].squeeze(0)):
        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // args.patch_size
        h_featmap = img.shape[-1] // args.patch_size

        attentions = model.get_last_selfattention(img.cuda())

        nh = attentions.shape[1] # number of head

        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
        
        if args.threshold is not None:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - 0.4)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = (
                nn.functional.interpolate(
                    th_attn.unsqueeze(0),
                    scale_factor=args.patch_size,
                    mode="nearest",
                )[0]
            )

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = (
            nn.functional.interpolate(
                attentions.unsqueeze(0),
                scale_factor=args.patch_size,
                mode="nearest",
            )[0]
        )

        if viz_atn:
            # save attentions heatmaps
            os.makedirs('attention_heads/', exist_ok=True)
            fig, axes = plt.subplots(1, nh+1, figsize=(20, 5))
            ax = axes[0]
            img_original = img.squeeze(0).permute(1,2,0).cpu().numpy()
            img_original = (img_original - img_original.min()) / (img_original.max() - img_original.min())
            img_original = (img_original * 255).astype(np.uint8)
            img_original = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
            ax.imshow(img_original)
            ax.set_title("Main Image")
            for j in range(1, nh+1):
                ax = axes[j]
                ax.imshow(attentions[j-1].cpu(), cmap="inferno")
                ax.set_title(f"Head {j-1}")

            fig.suptitle('Attention Heads Visualization')
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            output_path = os.path.join('attention_heads/', f"{i}_attn_heads.png")
            plt.savefig(output_path)
            print(f"{output_path} saved.")

        attention_list.append(attentions)
        
    concatenated_tensor = torch.stack(attention_list, dim=0)  # Shape: [8, 6, 224, 224] 
    all_attentions = concatenated_tensor.unsqueeze(1).unsqueeze(3).permute(2, 1, 0, 3, 4, 5)  # Shape: [6, 1, 8, 1, 224, 224]      

    return all_attentions

def save_video(frames, masked_img, output_path):
    # Select the first batch (assuming batch size is > 1)
    frames = frames[0][0]  # Shape: [num_frames, channels, height, width]
    frames = frames.permute(0, 2, 3, 1)  # Shape: [num_frames, height, width, channels]

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))
    
    # Ensure masked_img has the correct shape
    num_heads, batch_size, num_frames, channels, height_mask, width_mask = masked_img.shape
    assert batch_size == 1, "Batch size of masked_img should be 1"

    # Select 3 random heads
    choose_random_heads = [1,5,4] # random.sample(range(num_heads), 3)
    masked_img = masked_img[choose_random_heads, 0]  # Shape: [3, num_frames, channels, height_mask, width_mask]

    for frame_idx, frame in enumerate(frames):
        combined_frame = frame.cpu().numpy().copy()
        combined_frame = (combined_frame - combined_frame.min()) / (combined_frame.max() - combined_frame.min())
        combined_frame = (combined_frame * 255).astype(np.uint8)
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2GRAY)
        combined_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        
        # Create an empty attention map for 3 channels
        attn_map_combined = np.zeros((height, width, 3), dtype=np.uint8)
        
        for head_idx in range(3):
            attn_map = masked_img[head_idx, frame_idx].cpu().numpy()
            attn_map = cv2.resize(attn_map[0], (width, height))  # Resize attention map
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
            # attn_map * 1 / 3
            # attn_map = (attn_map > 0.5) * attn_map  # Apply threshold
            attn_map = (attn_map * 255).astype(np.uint8)
            attn_map = cv2.GaussianBlur(attn_map, (35, 35), 0)  # Apply Gaussian Blur
            attn_map_combined[:, :, head_idx] = attn_map  # Assign to the respective channel
        
        # Normalize combined attention map to [0, 255]
        attn_map_combined = (attn_map_combined - attn_map_combined.min()) / (attn_map_combined.max() - attn_map_combined.min())
        attn_map_combined = (attn_map_combined * 255).astype(np.uint8)

        # Convert combined attention map to BGR
        # attn_map_combined = cv2.cvtColor(attn_map_combined, cv2.COLOR_RGB2BGR)
        combined_frame = cv2.addWeighted(combined_frame, 0.3, attn_map_combined, 0.7, 0)
        
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
            # transforms.Resize((224,224)),
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
            global_1[cnt,:] = self.global_transfo1(gc_frame1[cnt,:])
            
            # global_2[cnt,:]= self.global_transfo2(gc_frame2[cnt,:])
        
        crops.append(global_1)
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
    masked_img = visualize_attention(model, frames, args.patch_size, viz_atn=False)
    save_video(frames, masked_img, args.output_video)
