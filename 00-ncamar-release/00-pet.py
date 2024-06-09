import sys
import numpy as np

from Misc.Utils import Unpickle,ReadImage
from Misc.Preview import Visualize3dImage
from Geometry.ExperimentalSetupPET import ExperimentalSetupPET
import matplotlib.pyplot as plt
from Algorithms.ProjectionDataGenerator import ProjectionDataGenerator 
from Algorithms.ART import ART
from Algorithms.SIRT import SIRT
from Algorithms.MLEM import MLEM
from Algorithms.OSEM import OSEM
from Misc.DataTypes import voxel_dtype
import matplotlib
import pdb
from Algorithms.SinogramGenerator import SinogramGenerator
import cv2
import time
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--network_init_img_path', type=str, default='imgs/01-MLEM/20.bin')
parse.add_argument('--img_noiseless_path', type=str, default='imgs/00-cleanImg/test.bin')
parse.add_argument('--intensity', type=int, default=10)
parse.add_argument('--save_path', type=str, default='IterativeCNN/')
parse.add_argument('--w_svd', type=float, default=0)
parse.add_argument('--onlyMLEM', type=int, default=0)
args = parse.parse_args()
args.img_name = args.img_noiseless_path.split('/')[-1][:-4]

# create an empty PET experimental setup
my_experimental_setup = ExperimentalSetupPET()
# radius of the cylindrical PET
my_experimental_setup.radius_mm = 128
# size of the pixels (visualization purpose only)
my_experimental_setup.pixel_size = np.array([1, 1, 1])
# number of pixels in the cylindrical geom
my_experimental_setup.pixels_per_slice_nb = 400
# number of detector's slice
my_experimental_setup.detector_slice_nb = 4
# pitch of the detector slices 
my_experimental_setup.slice_pitch_mm = 1
# fov size in mm 
my_experimental_setup.image_matrix_size_mm = np.array([128, 128, 4])
# voxel size in mm
my_experimental_setup.voxel_size_mm = np.array([1, 1, 1])
# h size of for the coincidences
my_experimental_setup.h_fan_size = 80
# (optional) give a name to the experimental setup 
my_experimental_setup.detector_name = "my first PET detector-128"

my_experimental_setup.Update()
print(my_experimental_setup.GetInfo())

try:
    input_img = np.fromfile(args.img_noiseless_path, np.float32).reshape(4, 128, 128).astype(voxel_dtype).transpose((1, 2, 0)) * 255
except:
    input_img = np.fromfile(args.img_noiseless_path, np.float64).reshape(4, 128, 128).astype(voxel_dtype).transpose((1, 2, 0)) * 255
cv2.imwrite('1-noiseless.jpg', (255-input_img[:, :, -1]).astype(np.uint8))
# pdb.set_trace()

g = ProjectionDataGenerator(my_experimental_setup)
# add noise to proj: 0 no noise added to projection 1 add poisson noise 2 add gaussian noise
noise = 1
projections = g.GenerateObjectProjectionData(input_img, args.img_name, noise, 0, args.intensity)


# algorithm must be one of "MLEM", "ART", "SIRT", "OSEM"
algorithm="MLEM"
# number of iterations 
niter= 100 #60
# number of subsets (OSEM only)
nsubsets=10
# when use using MLEM or OSEM remember to set this value to !=0 
initial_value=1



it = eval( algorithm + "()")
it.SetExperimentalSetup(my_experimental_setup)
it.SetNumberOfIterations(niter)
it.SetNumberOfSubsets(nsubsets)
it.SetProjectionData(projections)
it.SetArgs(args)

# start with a initial_guess filled image
initial_guess=np.full(it.GetNumberOfVoxels(), initial_value, dtype=voxel_dtype) 

it.SetImageGuess(initial_guess)
# uncomment this line to save images to disk
#it.SetOutputBaseName(basename)
if args.onlyMLEM:
    output_img = it.Reconstruct()
else:
    output_img = it.Reconstruct_DLMLEM()

