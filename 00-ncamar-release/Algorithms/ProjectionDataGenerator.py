import numpy as np
from Algorithms.IterativeReconstruction import IterativeReconstruction
import pdb
import random

class ProjectionDataGenerator:
    """!@brief
       Generate the LOR data of an image using the experimental_setup provided.  
       Projection data are stored as a 1d np array 
    """

    def __init__(self, experimental_setup):
        """!@brief
           Create the object and set the experimental setup needed to 
           evaluate the projections
        """
        self._experimental_setup = experimental_setup

    def GenerateObjectProjectionData(self, img, img_name, add_noise=0, transponse_image=0, intensity=20):
        """!@brief
            Generate the projection data of the an image 
            @param img: input image  
            @param add_noise: 0 no noise, 1 Poisson noise, 2 Gaussian noise  
            @param use_logarithm: whether to take the logarithm of the projection. Useful to simulate CT noise. 
            @param transponse_image: 1 to transpose x-y column of the image 0 otherwise
        """
        if transponse_image == 1:
            _img = np.transpose(img, axes=(1, 0, 2))
        else:
            _img = img
        # add an extra dim to account for z
        it = IterativeReconstruction()
        it.SetExperimentalSetup(self._experimental_setup)
        projections =  it.ForwardProjection(_img)
        # pdb.set_trace()
        # this transform the vector of projections
        # into a vector containing random Poisson distributed variables  with
        # average equal to the projection value
        if add_noise == 1:
            temp = projections.mean() / intensity #80
            projections = projections / temp
            projections = np.random.poisson(projections)
            projections = projections * temp

        # this transform the vector of projections
        # into a vector containing random Gaussian distributed variables  with
        # average equal to the projection value and std equal to the square root of the projections value
        if add_noise == 2:
            projections = np.random.normal(projections, np.sqrt(projections))
            # clip negative values to zero
            projections[projections < 0] = 0
        return projections
