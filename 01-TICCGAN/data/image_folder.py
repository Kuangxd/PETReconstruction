###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path
# import SimpleITK as sitk
import torch
import cv2
import numpy as np
import pdb

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.nii'
]

def load_nii_from_path(path):
    # load nii into tensor without change value
    # with self._open_file(path) as f:
    
    img = sitk.GetArrayFromImage(sitk.ReadImage(path))

    # 0: (3, 128, 128)
    # 3通道数据只需要一个,torch.Size([128, 128])
    img = torch.from_numpy(img)
    img = img.unsqueeze(0)
    return img

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_imageclip_dataset(dir, nframes, class_to_idx, vid_diverse_sampling, split='all'):
    """
    TODO: add xflip
    """
    def _sort(path):
        return sorted(os.listdir(path))
    
    images = []
    n_video = 0
    n_clip = 0

    dir_list = sorted(os.listdir(dir))
    for target in dir_list:
        if split == 'train':
            if 'val' in target: dir_list.remove(target)
        elif split == 'val' or split == 'test':
            if 'train' in target: dir_list.remove(target)

    for target in dir_list:
        if os.path.isdir(os.path.join(dir,target))==True:
            
            n_video +=1
            subfolder_path = os.path.join(dir, target)
            # for subsubfold in sorted(os.listdir(subfolder_path) ):
            if os.path.isdir(subfolder_path):
                
                # subsubfolder_path = os.path.join(subfolder_path, subsubfold)
                i = 1

                if nframes > 0 and vid_diverse_sampling:
                    n_clip += 1

                    item_frames_0 = []
                    item_frames_1 = []
                    item_frames_2 = []
                    item_frames_3 = []

                    for fi in _sort(subfolder_path):
                        if is_image_file(fi):
                            file_name = fi
                            file_path = os.path.join(subfolder_path, file_name)
                            item = (file_path, class_to_idx[target])

                            if i % 4 == 0:
                                item_frames_0.append(item)
                            elif i % 4 == 1:
                                item_frames_1.append(item)
                            elif i % 4 == 2:
                                item_frames_2.append(item)
                            else:
                                item_frames_3.append(item)

                            if i %nframes == 0 and i > 0:
                                images.append(item_frames_0) # item_frames is a list containing n frames.
                                images.append(item_frames_1) # item_frames is a list containing n frames.
                                images.append(item_frames_2) # item_frames is a list containing n frames.
                                images.append(item_frames_3) # item_frames is a list containing n frames.
                                item_frames_0 = []
                                item_frames_1 = []
                                item_frames_2 = []
                                item_frames_3 = []

                            i = i+1
                else:
                    item_frames = []
                    for fi in _sort(subfolder_path):

                        if is_image_file(fi):
                            # fi is an image in the subsubfolder
                            file_name = fi
                            file_path = os.path.join(subfolder_path, file_name)
                            item = (file_path, class_to_idx[target],target)
                            item_frames.append(item)
                            if i % nframes == 0 and i > 0:
                                images.append(item_frames)  # item_frames is a list containing 32 frames.
                                item_frames = []
                            i = i + 1

    return images

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

# def make_dataset_CT(dir):
#     classes, class_to_idx = find_classes(dir)
#     imgs = make_imageclip_dataset(dir, 4, class_to_idx, False)
    
#     return imgs

def make_dataset_CT(dir):
    imgs = []
    for root, _, files in os.walk(dir):
        for file in files:
            if not file.endswith('png'):
                continue
            img_path = os.path.join(root, file)
            imgs.append(img_path)
    
    return imgs

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

def make_dataset_PET(dir):
    imgs = []
    for root, _, files in os.walk(dir):
        for file in files:
            if 'noisehigh' in file:
                    continue
            img_path = os.path.join(root, file)
            imgs.append(img_path)
    
    return imgs

def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)
