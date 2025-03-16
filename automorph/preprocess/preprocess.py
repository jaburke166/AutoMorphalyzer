import os
import sys

SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))
MODULE_PATH = os.path.split(SCRIPT_PATH)[0]
PACKAGE_PATH = os.path.split(MODULE_PATH)[0]
sys.path.append(SCRIPT_PATH)
sys.path.append(MODULE_PATH)
sys.path.append(PACKAGE_PATH)

import torch
import pandas as pd
import timm
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import functional as F
import torch.nn.functional as TF
import preprocess.fundus_prep as prep
import utils


def get_QuickQual_model():
    device = utils.resolve_device()
    model = timm.create_model("densenet121.tv_in1k", pretrained=True, num_classes=0)
    model.eval().to(device)
    w = torch.tensor([-1411.32, 517.09, 342.41, -707.9,
                  1442.09, -23.25, -541.64, -8.44, 5.44])
    b = torch.tensor([5.18])

    return model, (w,b)


def get_quality(img, model_wb):
    device = utils.resolve_device()
    model, (w,b) = model_wb
    img = F.to_tensor(F.resize(Image.fromarray(img), 512))
    img = F.normalize(img, [0.5] * 3, [0.5] * 3).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = model(img).squeeze().cpu().reshape(1, -1)
    feats = feats[:, [71, 109, 121, 53, 55, 123, 29, 133, 84]]
    pred = torch.sigmoid(feats @ w + b)[0].item()

    return pred


def preprocess_dataset(input_path, image_list, save_path):

    # Initialise empty containers for information on preprocessed CFP
    radius_list = []
    centre_list_w = []
    centre_list_h = []
    name_list = []
    quality = []
    
    # Get QuickQual model
    model_wb = get_QuickQual_model()

    # Check if crop_info has already been made and load in if so
    crop_path = os.path.join(save_path, 'M0', 'crop_info.csv')
    if os.path.exists(crop_path):
        crop_df = pd.read_csv(crop_path)
    else:
        crop_df = None

    # Loop over images and pre-process and determine quality
    problem_log = []
    for image_path in tqdm(image_list, leave=False, desc='Pre-processing images'):        
        
        try:
            dst_image = os.path.join(input_path, image_path)
            if os.path.exists(f'{save_path}/M0/images/' + os.path.splitext(image_path)[0]+'.png'):
                if crop_df is not None:
                    crop_row = crop_df[crop_df.Name == os.path.splitext(image_path)[0]+'.png']
                    name_list.append(crop_row.Name.iloc[0])
                    centre_list_w.append(crop_row.centre_w.iloc[0])
                    centre_list_h.append(crop_row.centre_h.iloc[0])
                    radius_list.append(crop_row.radius.iloc[0])
                    quality.append(crop_row.quality.iloc[0])
                continue
                
            # Load in image and crop to square
            img = prep.imread(dst_image)
            r_img, borders, mask, r_img, radius_list, centre_list_w, centre_list_h = prep.process_without_gb(img,img,radius_list,centre_list_w, centre_list_h)
            
            # We resize to 912,912 to facilitate feature measurement and manual annotation
            imsize = (912,912)
            r_img = cv2.resize(np.float32(r_img), imsize, interpolation = cv2.INTER_NEAREST).astype(np.uint8)

            # Quality from QuickQual
            q = get_quality(r_img, model_wb)
            quality.append(q)

            # Write out image
            prep.imwrite(os.path.join(save_path, 'M0', 'images', image_path.split('.')[0] + '.png'), r_img)
            name_list.append(image_path.split('.')[0] + '.png')
        except Exception as e:
            msg = f"Error preprocessing {image_path}. Outputting error recieved:"
            print(msg)
            problem_log.append(msg)
            log = utils.print_error(e, verbose=True)
            problem_log.append(log) 
            continue

    # Save cropping info
    prep_df = pd.DataFrame({'Name':name_list, 'centre_w':centre_list_w, 'centre_h':centre_list_h, 'radius':radius_list, 'quality':quality})
    prep_df.to_csv(f'{save_path}/M0/crop_info.csv', index = None, encoding='utf8')

    # Save out log storing any problems
    if len(problem_log) > 0:
        with open(os.path.join(save_path, 'M3', 'preprocessing_log.txt'), 'w') as f:
            for item in problem_log:
                f.write("%s\n" % item)