# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
import numpy as np
from PIL import Image, ImageOps
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]


def normalize_rgb(img, imagenet_normalization=True):
    """
    Args:
        - img: np.array - (W,H,3) - np.uint8 - 0/255
    Return:
        - img: np.array - (3,W,H) - np.float - -3/3
    """
    img = img.astype(np.float32) / 255.
    img = np.transpose(img, (2,0,1))
    if imagenet_normalization:
        img = (img - np.asarray(IMG_NORM_MEAN).reshape(3,1,1)) / np.asarray(IMG_NORM_STD).reshape(3,1,1)
    img = img.astype(np.float32)
    return img

def denormalize_rgb(img, imagenet_normalization=True):
    """
    Args:
        - img: np.array - (3,W,H) - np.float - -3/3
    Return:
        - img: np.array - (W,H,3) - np.uint8 - 0/255
    """
    if imagenet_normalization:
        img = (img * np.asarray(IMG_NORM_STD).reshape(3,1,1)) + np.asarray(IMG_NORM_MEAN).reshape(3,1,1)
    img = np.transpose(img, (1,2,0)) * 255.
    img = img.astype(np.uint8)
    return img

def unpatch(data, patch_size=14, c=3, img_size=224):
    # c = 3
    if len(data.shape) == 2:
        c=1
        data = data[:,:,None].repeat([1,1,patch_size**2])

    B,N,HWC = data.shape
    HW = patch_size**2
    c = int(HWC / HW)
    h = w = int(N**.5)
    p = q = int(HW**.5)
    data = data.reshape([B,h,w,p,q,c])
    data = torch.einsum('nhwpqc->nchpwq', data)
    return data.reshape([B,c,img_size,img_size])



def open_image(img_path, img_size):
    """ Open image at path, resize and pad """

    # Open and reshape
    img_pil = Image.open(img_path).convert('RGB')
    
    # Get original size
    original_width, original_height = img_pil.size

    # reisze to the target size while keeping the aspect ratio
    img_pil = ImageOps.contain(img_pil, (img_size,img_size)) 

    # Get new size
    new_width, new_height = img_pil.size
    # Calculate scaling factors
    scale_x = original_width / new_width
    scale_y = original_height / new_height

    # Keep a copy for visualisations.
    img_pil_bis = ImageOps.pad(img_pil.copy(), size=(img_size,img_size), color=(255, 255, 255)) # image is keep centered
    img_pil = ImageOps.pad(img_pil, size=(img_size,img_size)) # pad with zero on the smallest side
    
    # Get new size
    padded_new_width, padded_new_height = img_pil_bis.size
    pad_width = (new_width - padded_new_width) / 2
    pad_height = (new_height - padded_new_height) / 2
    
    # Calculate translation
    translate_x = pad_width * scale_x
    translate_y = pad_height * scale_y
    
    # Create the affine transformation matrix
    affine_matrix = np.array([
        [scale_x, 0, translate_x],
        [0, scale_y, translate_y]
    ])

    # Go to numpy 
    resize_img = np.asarray(img_pil)

    # Normalize and go to torch.
    resize_img = normalize_rgb(resize_img)
    return resize_img, img_pil_bis, affine_matrix