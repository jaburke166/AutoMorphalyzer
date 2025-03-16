import os
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import random
from automorph.segment.optic_disc import paired_transforms_tv04 as p_tr


      
    
class AutomorphDataset(Dataset):
    def __init__(self, imgs_dir, model='binary', prev_fnames=None):
        self.imgs_dir = imgs_dir
        self.model = model
        if model == 'binary':
            self.img_size = (912,912)
        elif model == 'artery_vein':
            self.img_size = (720,720)
        elif model == 'optic_disc':
            self.img_size = (512,512)
        
        self.ids = list(Path(imgs_dir).glob('*.png'))
        self.ids = [os.path.split(path)[1] for path in self.ids]
        self.ids = [os.path.splitext(path)[0] for path in self.ids if not path.startswith('.')]
            
        logging.info(f'Number of total candidate files to process: {len(self.ids)}')
        if prev_fnames is not None:
            logging.info(f'Number of previously segmented files: {len(prev_fnames)}')
            self.ids =  [id for id in self.ids if id not in prev_fnames]
        logging.info(f'Creating dataset with {len(self.ids)} examples')


    def __len__(self):
        return len(self.ids)

    def preprocess(self, pil_img):

        img_array = np.array(pil_img)

        if self.model in ['binary', 'artery_vein']:
            mean=np.mean(img_array[img_array[...,0] > 0],axis=0)
            std=np.std(img_array[img_array[...,0] > 0],axis=0)

        if self.model == 'binary':
            img_array=(img_array-mean)/std
        elif self.model == 'artery_vein':
            img_array=(img_array-1.0*mean)/1.0*std # Incorrect standardisaton but kept for compatibility with model
        elif self.model == 'optic_disc':
            rsz = p_tr.Resize(self.img_size)
            tnsr = p_tr.ToTensor()
            tr = p_tr.Compose([rsz, tnsr])
            img = tr(pil_img)  
            return img

        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=2)

        img_array = img_array.transpose((2, 0, 1))

        return torch.from_numpy(img_array).type(torch.FloatTensor)


    def __getitem__(self, i):

        img_name = self.ids[i]
        img_file = os.path.join(self.imgs_dir,img_name+'.png')
        img = Image.open(img_file)
        ori_width, ori_height = img.size

        # Only resizing when artery_vein as pre-processed image is already (912,912) 
        # and optic disc is already resized in self.preprocess
        if self.model == 'artery_vein':
            img = img.resize(self.img_size)
        img = self.preprocess(img)
        
        return {
            'name': img_name,
            'width': ori_width,
            'height': ori_height,
            'image': img
        }

