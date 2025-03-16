import os
import sys

SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))
MODULE_PATH = os.path.split(SCRIPT_PATH)[0]
PACKAGE_PATH = os.path.split(MODULE_PATH)[0]
sys.path.append(SCRIPT_PATH)
sys.path.append(MODULE_PATH)
sys.path.append(PACKAGE_PATH)

from pathlib import Path
import dataset as automorph_dataset
from preprocess import fundus_prep as prep
import utils
from torch.utils.data import DataLoader
import torch.nn.functional as TF
from tqdm import tqdm
import torch
import numpy as np



def binary_vessel_segmentation(bin_networks, AUTOMORPH_RESULTS):

    # Path definitions
    test_dir = os.path.join(AUTOMORPH_RESULTS, 'M0', 'images')
    data_path = os.path.join(AUTOMORPH_RESULTS, 'M2', 'binary_vessel')
    results_dir = os.path.join(data_path, 'raw_binary')

    # Check to see which already has vessel segmentation
    prev_fpaths = list(set(list(Path(results_dir).glob('*.png'))).difference(set(list(Path(results_dir).glob('.*')))))
    prev_fnames = [os.path.splitext(os.path.split(p)[1])[0] for p in prev_fpaths]
    all_fpaths = list(set(list(Path(test_dir).glob('*.png'))).difference(set(list(Path(test_dir).glob('.*')))))
    if len(prev_fpaths) == len(all_fpaths):
        print('All images already previously segmented! Skipping.')
        return

    # Load binary vessel segmentation dataset
    batch_size = 4
    binary_dataset = automorph_dataset.AutomorphDataset(test_dir, model='binary', prev_fnames=prev_fnames)
    loader = DataLoader(binary_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)

    # Loop over batches and apply models, taken average sigmoid
    n_val = len(loader)
    all_bin_preds = []
    all_img_names = []
    N_bin_nets = len(bin_networks)
    device = utils.resolve_device()
    for batch in tqdm(loader, total=n_val, desc='Segmenting binary vessels', unit='batch', leave=True):
        
        # Unpack batch 
        binary_imgs = batch['image'].to(device=device, dtype=torch.float32)
        # ori_width=batch['width']
        # ori_height=batch['height']
        all_img_names.append(batch['name'])
    
        # For binary vessel segmentation
        # Inference per batch, extracting ensemble prediction for binary vessels 
        binary_pred_batch = 0
        # bin_preds = []
        with torch.no_grad():
            for net in bin_networks:
                pred_i = torch.sigmoid(net(binary_imgs))
                # bin_preds.append(pred_i)
                binary_pred_batch += pred_i
        binary_pred_batch /= N_bin_nets
    
        # Uncertainy map measures RMSE across all model predictions against the average
        # sigmoid prediction
        # uncertainty = torch.concat([torch.square(binary_pred - pred_i) for pred_i in binary_preds_all])
        # uncertainty = torch.sqrt(torch.sum(uncertainty, dim=0)/N_bin_nets)

        binmasks = utils.post_process_segs(binary_pred_batch, 
                             torch.tensor([912]), 
                             torch.tensor([912]), 
                             mode='binary')
        all_bin_preds.append(binmasks)

    # Prepare to save out
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    all_img_names = np.concatenate(all_img_names).reshape(-1,)
    all_bin_preds = np.concatenate(all_bin_preds).reshape(-1,912,912)

    # Save out
    N_imgs = all_img_names.shape[0]
    for name, img in tqdm(zip(all_img_names, all_bin_preds), total=N_imgs, desc='Saving binary segmentations', leave=False):
        prep.imwrite(os.path.join(results_dir, name + '.png'), (255*img).astype(np.uint8))


def arteryvein_vessel_segmentation(av_networks, AUTOMORPH_RESULTS):

    # Path definitions
    test_dir = os.path.join(AUTOMORPH_RESULTS, 'M0', 'images')
    data_path = os.path.join(AUTOMORPH_RESULTS, 'M2', 'artery_vein')
    results_dir = os.path.join(data_path, 'raw_binary')

    # Check to see which already has vessel segmentation
    prev_fpaths = list(set(list(Path(results_dir).glob('*.png'))).difference(set(list(Path(results_dir).glob('.*')))))
    prev_fnames = [os.path.splitext(os.path.split(p)[1])[0] for p in prev_fpaths]
    all_fpaths = list(set(list(Path(test_dir).glob('*.png'))).difference(set(list(Path(test_dir).glob('.*')))))
    if len(prev_fpaths) == len(all_fpaths):
        print('All images already previously segmented! Skipping.')
        return

    # Load binary vessel segmentation dataset
    batch_size = 4
    artery_vein_dataset = automorph_dataset.AutomorphDataset(test_dir, model='artery_vein', prev_fnames=prev_fnames)
    loader = DataLoader(artery_vein_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)

    # Loop over batches and apply models, taken average sigmoid
    av_preds_all =  []
    all_img_names = []
    N_av_nets = 8
    n_val = len(loader)
    device = utils.resolve_device()
    for batch in tqdm(loader, total=n_val, desc='Segmenting artery-veins', unit='batch', leave=True):
        # Unpack batch 
        av_imgs = batch['image'].to(device=device, dtype=torch.float32)
        # ori_width=batch['width']
        # ori_height=batch['height']
        all_img_names.append(batch['name'])
    
        # For artery-vein segmentation
        # Inference per batch, extracting ensemble prediction for artery-vein classification 
        av_pred_batch = 0
        # av_preds = []
        for model_i, model_dict in av_networks.items():
        
            with torch.no_grad():
                model_A, model_all, model_V = model_dict['A'], model_dict['all'], model_dict['V']
                _, masks_pred_G_fusion_A = model_A(av_imgs)
                _, masks_pred_G_fusion_V = model_V(av_imgs)
            
                mask_pred = model_all(av_imgs, 
                                    masks_pred_G_fusion_A.detach(), 
                                    masks_pred_G_fusion_V.detach())[0].clone().detach()
    
            # Run softmax on prediction across all 8 models and increment to final prediction
            av_pred_i = TF.softmax(mask_pred, dim=1).type(torch.FloatTensor)
            # av_preds.append(av_pred_i)
            av_pred_batch += av_pred_i
    
        #  Average over number of models and take maximum across all classification channels 
        # i.e. (background, artery, vein, crossings)
        av_pred_batch = (av_pred_batch/N_av_nets).to(device=device)
        av_pred_final = torch.max(av_pred_batch, 1)[1].type(torch.FloatTensor)
        if len(av_pred_final.size())==3:
            torch.unsqueeze(av_pred_final,0)
    
        # Uncertainy map measures RMSE across all model predictions against the average
        # prediction
        # uncertainty = torch.concat([torch.square(av_pred_all - pred_i) for pred_i in av_preds])
        # uncertainty = torch.sqrt(torch.sum(uncertainty, dim=0)/N_av_nets)

        avmasks = utils.post_process_segs(av_pred_final, 
                                          torch.tensor([912]),
                                          torch.tensor([912]),
                                          mode='artery_vein')   
        av_preds_all.append(avmasks)

    # Prepare to save out
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    all_img_names = np.concatenate(all_img_names).reshape(-1,)
    av_preds_all = np.concatenate(av_preds_all).reshape(-1,912,912)

    # Save out
    N_imgs = all_img_names.shape[0]
    for name, img in tqdm(zip(all_img_names, av_preds_all), total=N_imgs, desc='Saving artery-vein segmentations', leave=False):
        prep.imwrite(os.path.join(results_dir, name + '.png'), (255*img).astype(np.uint8))


def opticdisc_segmentation(od_networks, AUTOMORPH_RESULTS):

    # Path definitions
    test_dir = os.path.join(AUTOMORPH_RESULTS, 'M0', 'images')
    data_path = os.path.join(AUTOMORPH_RESULTS, 'M2', 'optic_disc')
    results_dir = os.path.join(data_path, 'raw_binary')

    # Check to see which already has vessel segmentation
    prev_fpaths = list(set(list(Path(results_dir).glob('*.png'))).difference(set(list(Path(results_dir).glob('.*')))))
    prev_fnames = [os.path.splitext(os.path.split(p)[1])[0] for p in prev_fpaths]
    all_fpaths = list(set(list(Path(test_dir).glob('*.png'))).difference(set(list(Path(test_dir).glob('.*')))))
    if len(prev_fpaths) == len(all_fpaths):
        print('All images already previously segmented! Skipping.')
        return

    # Load binary vessel segmentation dataset
    batch_size = 4
    optic_disc_dataset = automorph_dataset.AutomorphDataset(test_dir, model='optic_disc', prev_fnames=prev_fnames)
    loader = DataLoader(optic_disc_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=False, drop_last=False)

    # Loop over batches and apply models, taken average sigmoid
    n_val = len(loader)
    od_preds_all = []
    all_img_names = []
    device = utils.resolve_device()
    N_od_nets = len(od_networks)
    for batch in tqdm(loader, total=n_val, desc='Segmenting optic disc', unit='batch', leave=True):
        # Unpack batch 
        od_imgs = batch['image'].to(device=device, dtype=torch.float32)
        # ori_width=batch['width']
        # ori_height=batch['height']
        all_img_names.append(batch['name'])
    
        # For binary vessel segmentation
        # Inference per batch, extracting ensemble prediction for binary vessels 
        od_pred_batch = 0
        with torch.no_grad():
            # od_preds = []
            for net in od_networks:
                mask_pred = TF.softmax(net(od_imgs)[1].clone().detach(), dim=1)
                # od_preds.append(mask_pred)
                od_pred_batch += mask_pred.type(torch.FloatTensor)  
    
        #  Average over number of models and take maximum across all classification channels 
        # i.e. (background, disc, cup)
        od_pred_batch = (od_pred_batch/N_od_nets).to(device=device)
        od_pred_final = torch.max(od_pred_batch, 1)[1].type(torch.FloatTensor)
        if len(od_pred_final.size())==3:
            torch.unsqueeze(od_pred_final,0)
        
        # Uncertainy map measures RMSE across all model predictions against the average
        # # prediction
        # uncertainty = torch.concat([torch.square(od_pred_all - pred_i) for pred_i in od_preds])
        # uncertainty = torch.sqrt(torch.sum(uncertainty, dim=0)/N_av_nets)
        
        odmasks = utils.post_process_segs(od_pred_final, 
                                  torch.tensor([912]),
                                  torch.tensor([912]), 
                                  mode='optic_disc')   
        od_preds_all.append(odmasks)

    # Prepare to save out
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    all_img_names = np.concatenate(all_img_names).reshape(-1,)
    od_preds_all = np.concatenate(od_preds_all).reshape(-1,912,912)

    # Save out
    N_imgs = all_img_names.shape[0]
    for name, img in tqdm(zip(all_img_names, od_preds_all), total=N_imgs, desc='Saving optic disc segmentations', leave=False):
        prep.imwrite(os.path.join(results_dir, name + '.png'), (255*img).astype(np.uint8))


        