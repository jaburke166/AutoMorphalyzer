import os
import sys

SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))
MODULE_PATH = os.path.split(SCRIPT_PATH)[0]
PACKAGE_PATH = os.path.split(MODULE_PATH)[0]
sys.path.append(SCRIPT_PATH)
sys.path.append(MODULE_PATH)
sys.path.append(PACKAGE_PATH)
SEGMENTATION_WEIGHT_PATH = os.path.join(SCRIPT_PATH, 'segment')

import torch
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from skimage import measure, morphology
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import requests

from segment.binary.model import Segmenter
from segment.artery_vein.model import Generator_main, Generator_branch
from segment.optic_disc.models import get_model



def _download_zipfile(url, filepath):
    
    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)

    # Sizes in bytes.
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(filepath, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")



def check_unpack_model_weights(model_type, save_path):

    if model_type == 'binary':
        N_model = 10
        zip_url = 'https://github.com/jaburke166/AutoMorphalyzer/releases/download/v1.0/binary_model_weights.zip'
    elif model_type == 'artery_vein':
        N_model = 24
        zip_url = 'https://github.com/jaburke166/AutoMorphalyzer/releases/download/v1.0/arteryvein_model_weights.zip'
    elif model_type == 'optic_disc':
        N_model = 8
        zip_url = 'https://github.com/jaburke166/AutoMorphalyzer/releases/download/v1.0/opticdisc_model_weights.zip'

    # Destination for zip and model weights
    destination_folder = os.path.join(save_path, 'segment', model_type)   
    zip_fname = os.path.split(zip_url)[1]
    zip_path = os.path.join(destination_folder, zip_fname)

    # Check for model weights
    weight_list = list(Path(os.path.join(destination_folder, 'model_weights')).glob('*.pth'))
    if len(weight_list) != N_model:
        print(f"Downloading {zip_url} to {destination_folder}")

        # Download the zip file
        _download_zipfile(zip_url, zip_path)

        # Extract the zip file
        os.makedirs(destination_folder, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for member in tqdm(zip_ref.infolist(), desc='Extracting ', total=N_model, leave=False, unit='model'):
                 zip_ref.extract(member, destination_folder)

        # Remove downloaded .zip file
        os.remove(zip_path)

    else:
        print(f'{model_type.capitalize()} model weights already downloaded!')



def resolve_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    return device



def get_binary_models():
    MODEL_PATH = os.path.join(SEGMENTATION_WEIGHT_PATH,
                              'binary', 
                              'model_weights')
    model_paths = list(Path(MODEL_PATH).glob('*.pth'))    
    device = resolve_device()
    networks = []
    for path in tqdm(model_paths, desc='Loading binary vessel segmentation models', total=len(model_paths), unit='model', leave=True):
        model = Segmenter(input_channels=3, n_filters = 32, n_classes=1, bilinear=False)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        model.to(device=device)
        networks.append(model)

    return networks



def get_av_models():
    MODEL_PATH = os.path.join(SEGMENTATION_WEIGHT_PATH, 
                              'artery_vein', 
                              'model_weights')
    
    # Loop across number of models, loading the Artery-Vein branches and entire generator models. 
    # 3 networks per model. 8 models overall. Stored in dictionary for easy look-up
    device = resolve_device()
    av_networks = {}
    N_av_nets = 8
    for i in tqdm(range(1,N_av_nets+1), desc='Loading artery-vein segmentation models', total=N_av_nets, unit='model', leave=True):
    
        model_path = os.path.join(MODEL_PATH, f'model_{i}')
        av_networks[i] = {}
        for typ in ['A', 'all', 'V']:
            if typ == 'all':
                model = Generator_main(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
            else:
                model = Generator_branch(input_channels=3, n_filters = 32, n_classes=4, bilinear=False)
            model.load_state_dict(torch.load(model_path+f'{typ}.pth', map_location=device))
            model.eval()
            model.to(device=device)
            av_networks[i][typ] = model

    return av_networks



def get_od_models():
    MODEL_PATH = os.path.join(SEGMENTATION_WEIGHT_PATH,
                              'optic_disc', 
                              'model_weights')
    model_paths = list(Path(MODEL_PATH).glob('*.pth'))
    
    device = resolve_device()
    od_networks = []
    for path in tqdm(model_paths, desc='Loading optic disc segmentation models', unit='model', leave=True):
        model = get_model.get_arch("wnet", n_classes=3).to(device)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        od_networks.append(model)

    return od_networks
 


def post_process_segs(preds, height, width, mode='binary'):

    n_img = preds.shape[0]
    masks = []
    for i in range(n_img):
        pred_i = preds[i].cpu().detach().numpy()
        imsize = (height.item(), width.item())
        if mode == 'binary':
            pred_i = pred_i.squeeze()
            mask_i = (pred_i>=0.5).astype(np.uint8)
        elif mode == 'artery_vein':
            pred_a, pred_v, pred_c = pred_i==1, pred_i==2, pred_i==3
            pred_a = morphology.remove_small_objects(pred_a, 30, connectivity=5)
            pred_v = morphology.remove_small_objects(pred_v, 30, connectivity=5)
            mask_i = np.concatenate([pred_a[...,np.newaxis], 
                                         pred_c[...,np.newaxis], 
                                         pred_v[...,np.newaxis]], axis=2)
        elif mode == 'optic_disc':
            pred_d, pred_c = pred_i==1, pred_i==2
            pred_c = morphology.remove_small_objects(pred_c, 50)
            pred_d = morphology.remove_small_objects(pred_d, 100)
            mask_i = np.concatenate([pred_d[...,np.newaxis], 
                                         pred_c[...,np.newaxis], 
                                         np.zeros_like(pred_d)[...,np.newaxis]], axis=2)

        mask_i = cv2.resize(np.float32(mask_i), imsize, interpolation = cv2.INTER_NEAREST)
        masks.append(mask_i.astype(np.uint8))

    return np.asarray(masks)  



def _generate_imgmask(mask, thresh=None, cmap=0):
    '''
    Given a prediction mask Returns a plottable mask
    '''
    # Threshold
    pred_mask = mask.copy()
    if thresh is not None:
        pred_mask[pred_mask < thresh] = 0
        pred_mask[pred_mask >= thresh] = 1
    max_val = pred_mask.max()
    
    # Compute plottable cmap using transparency RGBA image.
    trans = max_val*((pred_mask > 0).astype(int)[...,np.newaxis])
    if cmap is not None:
        rgbmap = np.zeros((*mask.shape,3))
        rgbmap[...,cmap] = pred_mask
    else:
        rgbmap = np.transpose(3*[pred_mask], (1,2,0))
    pred_mask_plot = np.concatenate([rgbmap,trans], axis=-1)
    
    return pred_mask_plot



def _fit_ellipse(mask, get_contours=False):

    # fit minimum area ellipse around disc
    _, thresh = cv2.threshold(mask, 127, 255, 1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    if get_contours:
        return cnt
    ellipse = cv2.fitEllipse(cnt)
    new_mask = cv2.ellipse(np.zeros_like(mask), ellipse, (255,255,255), -1)/255

    return new_mask



def process_opticdisc(od_mask):
    """
    Work out optic disc radius in pixels, according to it's position relative to the fovea.
    """
    # Extract Optic disc radius and OD boundary if detected
    try:
        od_mask_props = measure.regionprops(measure.label(od_mask))[0]
    except:
        return None, np.zeros_like(od_mask)
    od_radius = int((od_mask_props.axis_minor_length + od_mask_props.axis_major_length)/4)
    # od_boundary = segmentation.find_boundaries(od_mask)

    return od_radius



def _create_circular_mask(center, img_shape, radius):
    """
    Given a center, radius and image shape, draw a filled circle
    as a binary mask.
    """
    # Circular mask
    h, w = img_shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = (dist_from_center <= radius).astype(int)
    
    return mask



def generate_zonal_masks(img_shape, od_radius, od_centre):

    mask_rois = {}
    for roi_type in ['whole', 'B', 'C']:
        if roi_type == 'whole':
            mask = np.ones(img_shape)
        else:
            od_diameter = 2*od_radius
            if roi_type == "B":
                od_circ = _create_circular_mask(img_shape=img_shape, 
                                            radius=2*od_radius, 
                                            center=od_centre)[0]
                
                mask  = _create_circular_mask(img_shape=img_shape, 
                                                    radius=3*od_radius, 
                                                    center=od_centre)
                macula_p = 3*od_diameter
            elif roi_type == "C":
                od_circ = _create_circular_mask(img_shape=img_shape, 
                                            radius=2*od_radius, 
                                            center=od_centre)[0]
                
                mask = _create_circular_mask(img_shape=img_shape, 
                                                radius=5*od_radius, 
                                                center=od_centre)
                macula_p = 5*od_diameter

            mask -= od_circ
        mask_rois[roi_type] = mask

    return mask_rois



def superimpose_segmentations(cfp_img, binmask, avmask, disc_mask, cup_mask, output_directory, img_name):
    
    img_size = cfp_img.shape[0]
    stacked_img = np.hstack(3*[cfp_img])
    avmask = np.concatenate([avmask, np.sum(avmask > 0, axis=-1)[...,np.newaxis]], axis=-1)
    cfp_vcmap = _generate_imgmask(binmask, None, 1)
    stacked_cmap = np.hstack([np.zeros_like(cfp_vcmap), cfp_vcmap, avmask])
    
    disc_coords = _fit_ellipse(disc_mask, get_contours=True)[:,0]
    cup_coords = _fit_ellipse(cup_mask, get_contours=True)[:,0]
    od_coords = disc_coords[(disc_coords[:,0] > 0) & (disc_coords[:,0] < img_size-1)]
    oc_coords = cup_coords[(cup_coords[:,0] > 0) & (cup_coords[:,0] < img_size-1)]
    
    fig, ax = plt.subplots(1,1,figsize=(18,6))
    ax.imshow(stacked_img, cmap="gray")
    ax.imshow(stacked_cmap, alpha=0.5)
    for i in [img_size, 2*img_size]:
        if i == 2*img_size:  
            ax.plot(od_coords[:,0]+i, od_coords[:,1], color='lime', linestyle='--', linewidth=3, zorder=4)
            ax.plot(oc_coords[:,0]+i, oc_coords[:,1], color='lime', linestyle='--', linewidth=3, zorder=4)
        else:
            ax.plot(od_coords[:,0]+i, od_coords[:,1], color='blue', linestyle='--', linewidth=3, zorder=4)
            ax.plot(oc_coords[:,0]+i, oc_coords[:,1], color='blue', linestyle='--', linewidth=3, zorder=4)
    ax.set_axis_off()
    fig.tight_layout(pad = 0)
    fig.savefig(os.path.join(output_directory, img_name + '.png'), bbox_inches="tight")
    plt.close()

    

def _flatten_dict(nested_dict):
    '''
    Recursive flattening of a dictionary of dictionaries.
    '''
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = _flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res



def nested_dict_to_df(values_dict):
    '''
    Nested dictionary is flattened and converted into an index-wise, multi-level Pandas DataFrame
    '''
    flat_dict = _flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map("{0[1]}".format)
    return df



def select_largest_mask(binmask):
    """
    Retain only the largest connected region in a binary mask.

    Parameters:
    -----------
    binmask : numpy.ndarray
        Binary mask with (potentially) multiple connected regions.

    Returns:
    --------
    numpy.ndarray
        Binary mask with only the largest connected region retained.
    """
    # Look at which of the region has the largest area, and set all other regions to 0
    labels_mask = measure.label(binmask)                       
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1

    return labels_mask



def print_error(e, verbose=True):
    '''
    If an unexpected error occurs, this will be printed out and saved to a log.

    A detailed explanation of the error found.
    '''
    message = f"\nAn exception of type {type(e).__name__} occurred. Error description:\n{str(e)}\n"
    if verbose:
        print(message)
    trace = ["Full traceback:\n"]
    if verbose:
        print(trace[0])
    tb = e.__traceback__
    tb_i = 1
    while tb is not None:
        tb_fname = tb.tb_frame.f_code.co_filename
        tb_func = tb.tb_frame.f_code.co_name
        tb_lineno = tb.tb_lineno
        tb_str = f"Traceback {tb_i} to filename\n{tb_fname}\nfor function {tb_func}(...) at line {tb_lineno}.\n"
        if verbose:
            print(tb_str)
        trace.append(tb_str)
        tb = tb.tb_next
        tb_i += 1
    logging_list = [message] + trace
    return logging_list