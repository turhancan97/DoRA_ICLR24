import sys
sys.path.append('model/')

import torchinfo
import torch

def summary(model, batch_size):
    # Torchvision Faster RCNN models are enclosed within a tuple ().
    if type(model) == tuple:
        model = model[0]
    device = 'cpu'
    batch_size = batch_size
    channels = 3
    img_height = 256
    img_width = 256
    torchinfo.summary(
        model, 
        device=device, 
        input_size=[batch_size, channels, img_height, img_width],
        row_settings=["var_names"]
    )