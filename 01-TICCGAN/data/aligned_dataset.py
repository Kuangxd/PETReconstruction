import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset_PET
from data.image_folder import load_nii_from_path
from PIL import Image
# import SimpleITK as sitk
import pdb
import cv2
import random
import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter, gaussian_gradient_magnitude
from random import sample

def make_dataset_PET():
    img_noiseless_path_list = []
    img_noisehigh_path_list = []


    img_noiseless_root = ''
    img_names = os.listdir(img_noiseless_root)
    
    img_noisehigh_root = ''
    img_noisehigh_list = os.listdir(img_noisehigh_root)

    for eachImg in img_names:
        if eachImg[:-4] in img_noisehigh_list:
            for i in range(20, 61, 5):
                if os.path.isfile(os.path.join(img_noisehigh_root, eachImg[:-4], str(i)+'.bin')) and os.stat(os.path.join(img_noisehigh_root, eachImg[:-4], str(i)+'.bin')).st_size > 0:
                    img_noiseless_path_list.append(os.path.join(img_noiseless_root, eachImg))
                    img_noisehigh_path_list.append(os.path.join(img_noisehigh_root, eachImg[:-4], str(i)+'.bin'))
    # pdb.set_trace()
    return img_noiseless_path_list, img_noisehigh_path_list

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.isTrain = opt.isTrain

        self.img_noiseless_path_list, self.img_noisehigh_path_list = make_dataset_PET()
        self.AB_paths = self.img_noiseless_path_list    

    def __getitem__(self, index):
        
        img_noiseless_path = self.img_noiseless_path_list[index]
        img_noisehigh_path = self.img_noisehigh_path_list[index]
        

        try:
            img_lowNoise = np.fromfile(img_noiseless_path, dtype=np.float64).reshape(4, 128, 128)
        except:
            img_lowNoise = np.fromfile(img_noiseless_path, dtype=np.float32).reshape(4, 128, 128)
        try:
            img_highNoise = np.fromfile(img_noisehigh_path, dtype=np.float64).reshape(4, 128, 128)
        except:
            img_highNoise = np.fromfile(img_noisehigh_path, dtype=np.float32).reshape(4, 128, 128)

        img_lowNoise = torch.from_numpy(img_lowNoise)
        img_lowNoise = img_lowNoise.float()

        img_highNoise = torch.from_numpy(img_highNoise)
        img_highNoise = img_highNoise.float()

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        
        return {'A': img_highNoise , 'B': img_lowNoise,
                'A_paths': img_noiseless_path, 'B_paths': img_noiseless_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
