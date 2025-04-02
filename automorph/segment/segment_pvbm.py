# Code from L3-111 comes directly from PVBM python package 
# Script webpage: https://github.com/aim-lab/PVBM/blob/main/PVBM/DiscSegmenter.py
# Author: Johnathan Fhima
# Modified for AutoMorphalyzer

import os
import sys

SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))
MODULE_PATH = os.path.split(SCRIPT_PATH)[0]
PACKAGE_PATH = os.path.split(MODULE_PATH)[0]
sys.path.append(SCRIPT_PATH)
sys.path.append(MODULE_PATH)
sys.path.append(PACKAGE_PATH)

import os
import onnxruntime as ort
import numpy as np
import PIL
from torchvision import transforms
import cv2
import tensorflow as tf
from automorph.segment.artery_vein.PVBM import model
from automorph import utils
from pathlib import Path
from tqdm import tqdm
from preprocess import fundus_prep as prep
import logging

class DiscSegmenter:
    def __init__(self):
        self.img_size = 512
        self.model_path = os.path.join(SCRIPT_PATH, 'optic_disc', 'pvbm_weights', 'lunetv2_odc.onnx')

    def segment(self, img_orig, target_size=912):
        session = ort.InferenceSession(self.model_path)
        input_name = session.get_inputs()[0].name

        img_orig = PIL.Image.open(img_orig)
        image = img_orig.resize((self.img_size, self.img_size))
        image = transforms.ToTensor()(image)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)

        image_np = image.numpy()
        image_np = np.expand_dims(image_np, axis=0)

        outputs = session.run(None, {input_name: image_np})
        od = outputs[0][0, 0] > 0

        # Resize 
        od = PIL.Image.fromarray(np.array(od, dtype=np.uint8) * 255).resize((target_size, target_size), PIL.Image.Resampling.NEAREST)

        return np.array(od)
        

class VesselSegmenter():
    def __init__(self):
        self.input_size = 1472
        self.target_size = 1444
        self.model_path = os.path.join(SCRIPT_PATH, 'artery_vein', 'pvbm_weights', 'lunet_modelbest.h5')

        # Set up GPU acceleration if possible
        tf.debugging.set_log_device_placement(True)
        physical_devices = tf.config.list_physical_devices('GPU')
        if tf.config.list_physical_devices('GPU'):
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            except:
                pass
        self.get_model()
        

    def get_model(self):

        # Initialising model
        inputs = tf.keras.layers.Input((self.input_size, self.input_size,3))
        init_n_filters = 18
        kernel_size = 7
        transpose_stride = 2
        stride = 1
        keep_prob = 0.9
        block_size = 7
        drop_block = True
        with_batch_normalization = True
        self.model, _, _ = model.build_lunet(inputs,
                                        init_n_filters,
                                        kernel_size,
                                        transpose_stride,
                                        stride,
                                        keep_prob,
                                        block_size,
                                        with_batch_normalization = with_batch_normalization, 
                                        scale = True,
                                        dropblock = drop_block)
        
        # Load model
        self.model.load_weights(self.model_path)


    def segment(self, batch, target_size=912):

        # Run inference
        pred = self.model(batch, training=False)

        # Remove the padding added to the image anbd resize to common resolution for save out
        # and feature measurement
        pred = tf.image.resize(pred[:,14:-14,14:-14],(target_size, target_size)).numpy()

        # Threshold and collect
        binmap = pred > 0
        binary = binmap.sum(axis=-1)
        artery = binmap[...,1]
        vein = binmap[...,0]
        crossing = artery * vein

        # Post-process as binary maps per image
        all_binary = []
        all_avmaps = []
        for bina, art, vei, cro in zip(binary, artery, vein, crossing):
            bina = utils.process_vesselmap(bina)
            art = utils.process_vesselmap(art)
            vei = utils.process_vesselmap(vei)

            # Remove crossings to save out
            art = art.astype(bool) ^ cro
            vei = vei.astype(bool) ^ cro

            # Collect artery, vein and crossings
            av_map = (127/255)*art + (191/255)*vei + (255/255)*cro
            all_avmaps.append(av_map)
            all_binary.append(bina)

        # Concatenate all results and return
        all_avmaps = np.asarray(all_avmaps)
        all_binary = np.asarray(all_binary)

        return all_binary, all_avmaps
    

# Prepare for processing
def load_and_preprocess_image(image_path, target_size=1444):
    image = tf.io.read_file(image_path)

    def decode_bmp(): return tf.image.decode_bmp(image, channels=3)
    def decode_jpg(): return tf.image.decode_jpeg(image, channels=3)
    def decode_png(): return tf.image.decode_png(image, channels=3)
    
    # Convert Tensor string to lowercase
    image_lower = tf.strings.lower(image_path)

    # Use tf.case() to choose the right decoder
    image = tf.case([
        (tf.strings.regex_full_match(image_lower, ".*\\.bmp"), decode_bmp),
        (tf.strings.regex_full_match(image_lower, ".*\\.jpg"), decode_jpg),
        (tf.strings.regex_full_match(image_lower, ".*\\.jpeg"), decode_jpg),
        (tf.strings.regex_full_match(image_lower, ".*\\.png"), decode_png),
    ], exclusive=True)

    # Resize, pad and normalise
    image = tf.image.resize(image, [target_size, target_size]) 
    image = tf.image.pad_to_bounding_box(image, 14, 14, 1472, 1472)
    image = tf.cast(image, tf.float32) / 255.0
    return image
    

def vessel_segmentation(AUTOMORPH_RESULTS):

    # Path definitions
    test_dir = os.path.join(AUTOMORPH_RESULTS, 'M0', 'images')
    bindata_path = os.path.join(AUTOMORPH_RESULTS, 'M2', 'binary_vessel')
    avdata_path = os.path.join(AUTOMORPH_RESULTS, 'M2', 'artery_vein')
    bin_results_dir = os.path.join(bindata_path, 'raw_binary')
    av_results_dir = os.path.join(avdata_path, 'raw_binary')
    
    # Check to see which already has a binary vessel segmentation and use this as a surrogate for what files are left
    # to be processed
    prev_fpaths = list(set(list(Path(bin_results_dir).glob('*.png'))).difference(set(list(Path(bin_results_dir).glob('.*')))))
    prev_fnames = [os.path.splitext(os.path.split(p)[1])[0] for p in prev_fpaths]
    all_fpaths = list(set(list(Path(test_dir).glob('*.png'))).difference(set(list(Path(test_dir).glob('.*')))))
    all_fnames = [os.path.splitext(os.path.split(p)[1])[0] for p in all_fpaths]
    if len(prev_fpaths) == len(all_fpaths):
        print('All images already previously segmented! Skipping.')
        return
    
    # work out remaining fpaths to process
    rem_fnames = list(set(all_fnames).difference(set(prev_fnames)))
    rem_fpaths = [str(os.path.join(test_dir, fname+'.png')) for fname in rem_fnames]
    logging.info(f'Number of total candidate files to process: {len(all_fnames)}')
    logging.info(f'Number of previously segmented files: {len(prev_fnames)}')
    logging.info(f'Creating dataset with {len(rem_fnames)} examples.')

    # Load binary vessel segmentation dataset
    batch_size = 4
    AUTOTUNE = tf.data.AUTOTUNE  
    loader = (tf.data.Dataset.from_tensor_slices(rem_fpaths)
            .map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE) 
            .batch(batch_size)  # Efficient batching
            .prefetch(AUTOTUNE)) 
    
    # Instantiate model
    model = VesselSegmenter()

    # Loop over batches and apply models, taken average sigmoid
    n_val = len(loader)
    all_bin_preds = []
    all_av_preds = []
    for batch in tqdm(loader, total=n_val, desc='Segmenting vessels', unit='batch', leave=True):
        
        # Unpack batch 
        batch_binpred, batch_avpred = model.segment(batch, 912)
        all_bin_preds.append(batch_binpred)
        all_av_preds.append(batch_avpred)

    # Collect predictions and filenames
    all_img_names = np.asarray(rem_fnames).reshape(-1,)
    all_bin_preds = np.concatenate(all_bin_preds).reshape(-1,912,912)
    all_av_preds = np.concatenate(all_av_preds).reshape(-1,912,912)

    # Prepare to save out
    N_imgs = all_img_names.shape[0]
    if not os.path.exists(bin_results_dir):
        os.makedirs(bin_results_dir)
    if not os.path.exists(av_results_dir):
        os.makedirs(av_results_dir)
    for name, bin_img, av_img, in tqdm(zip(all_img_names, all_bin_preds, all_av_preds), 
                                       total=N_imgs, desc='Saving vessel segmentations', leave=True):
        prep.imwrite(os.path.join(bin_results_dir, name + '.png'), (255*bin_img).astype(np.uint8))
        prep.imwrite(os.path.join(av_results_dir, name + '.png'), (255*av_img).astype(np.uint8))



def opticdisc_segmentation(AUTOMORPH_RESULTS):

    # Path definitions
    test_dir = os.path.join(AUTOMORPH_RESULTS, 'M0', 'images')
    data_path = os.path.join(AUTOMORPH_RESULTS, 'M2', 'optic_disc')
    results_dir = os.path.join(data_path, 'raw_binary')
    
    # Check to see which already has a binary vessel segmentation and use this as a surrogate for what files are left
    # to be processed
    prev_fpaths = list(set(list(Path(results_dir).glob('*.png'))).difference(set(list(Path(results_dir).glob('.*')))))
    prev_fnames = [os.path.splitext(os.path.split(p)[1])[0] for p in prev_fpaths]
    all_fpaths = list(set(list(Path(test_dir).glob('*.png'))).difference(set(list(Path(test_dir).glob('.*')))))
    all_fnames = [os.path.splitext(os.path.split(p)[1])[0] for p in all_fpaths]
    if len(prev_fnames) == len(all_fnames):
        print('All images already previously segmented! Skipping.')
        return
    
    # work out remaining fpaths to process
    rem_fnames = list(set(all_fnames).difference(set(prev_fnames)))
    rem_fpaths = [os.path.join(test_dir, fname+'.png') for fname in rem_fnames]
    logging.info(f'Number of total candidate files to process: {len(all_fnames)}')
    logging.info(f'Number of previously segmented files: {len(prev_fnames)}')
    logging.info(f'Creating dataset with {len(rem_fnames)} examples.')

    # Instantiate model
    model = DiscSegmenter()

    # Loop over batches and apply models, taken average sigmoid
    n_val = len(rem_fnames)
    all_od_preds = []
    for impath in tqdm(rem_fpaths, total=n_val, desc='Segmenting optic disc', unit='image', leave=True):
        
        # Unpack batch 
        od_pred = model.segment(impath, 912)
        all_od_preds.append(od_pred)

    # Collect predictions and filenames
    all_img_names = np.asarray(rem_fnames).reshape(-1,)
    all_od_preds = np.concatenate(all_od_preds).reshape(-1,912,912)

    # Prepare to save out
    N_imgs = all_img_names.shape[0]
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    for name, od_img, in tqdm(zip(all_img_names, all_od_preds), 
                                       total=N_imgs, desc='Saving vessel segmentations', leave=True):
        prep.imwrite(os.path.join(results_dir, name + '.png'), od_img)

