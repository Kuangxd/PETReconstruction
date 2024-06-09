import numpy as np
from Algorithms.SiddonProjector import SiddonProjector
import sys
import time
from Misc.DataTypes import voxel_dtype, projection_dtype
from Misc.Utils import Pickle
import matplotlib
import os
import pdb
import cv2

projector_debug_msg =0


class IterativeReconstruction:
    """!@brief
         Implements all the basic operations needed by iterative reconstruction algorithms.
    """
    
    def __init__(self):
        self._save_img_to_disk = 0
        self._output_file_name = ""

    def SetExperimentalSetup(self, experimental_setup):
        """!@brief  
        Load  an experimental setup
        @param experimental_setup: the object representing the experimental setup  
        """
        self._experimental_setup = experimental_setup
        self._SiddonProjector = SiddonProjector(
            self._experimental_setup.image_matrix_size_mm,
            self._experimental_setup.voxel_size_mm,
        )

    def GetNumberOfVoxels(self):
        """!@brief 
        Return the number of voxels along each direction of the  as np.array
        """
        return self._SiddonProjector._voxel_nb

    def GetNumberOfProjections(self):
        """!@brief 
        Return the number of projections defined in the experimental setup
        """
        return self._experimental_setup._number_of_projections

    def SetImageGuess(self, image):
        """!@brief 
        Set initial guess image for starting the iterative procedure 
        """
        self._image = image.astype(voxel_dtype)
    
    def SetArgs(self, args):
        """!@brief 
        Set initial guess image for starting the iterative procedure 
        """
        self.args = args

    def SetNumberOfIterations(self, niter):
        """!@brief 
        Set the number of iterations to be performed
        @param  niter:  number of iteration to be performed by the iterative algorithm
        """
        self._niter = niter

    def SetNumberOfSubsets(self, subsetnumber):
        """!@brief 
        Set the number of data subsets. This option is for OSEM only.
        @param subsetnumber: number of subsets
        """
        self._subsetnumber = subsetnumber

    def SetOutputBaseName(self, output_file_name):
        """!@brief
        Set the base name for saving the output images. 
        By default the images are not saved to disk unless a base name is set.
        """
        self._save_img_to_disk = 1
        self._output_file_name = output_file_name

    def SetProjectionData(self, my_data):
        """!@brief
        Load a dense np.array containing the projection to be reconstructed 
        @param my_data: np.array containing the projection to be reconstructed 
        """
        self._projection_data = my_data
        # if length is different from number of LORs
        #
        if len(self._projection_data) != self.GetNumberOfProjections():
            print(
                "Something wrong with the input data: expected {} TORs got {}".format(
                    self.GetNumberOfProjections(), len(self._projection_data)
                )
            )
            sys.exit()

    def ComputeTOR(self, tor_id):
        """!@brief
        Compute the Tube Of Response relative to the tor_id-th projection using the  Siddon projector
        and return it as np.array of  TOR_dtype
        @param  tor_id: TOR id according of the loaded experimental setup 
        """
        p0 = self._experimental_setup._projections_extrema["p0"][tor_id]
        p1 = self._experimental_setup._projections_extrema["p1"][tor_id]
        TOR = self._SiddonProjector.CalcIntersection(p0, p1)
        return TOR

    def ForwardProjectSingleTOR(self, img, tor_id):
        """!@brief
            Perform the forward projection of the tor_id-th TOR on the image 
            @param img:     image used to perform the projection
            @param tor_id:  TOR id according of the loaded experimental setup 
        """
       
        try:
            TOR = self.ComputeTOR(tor_id)
            # pdb.set_trace()
            return np.sum(img[TOR["vx"], TOR["vy"], TOR["vz"]] * TOR["prob"])
        except:
            if projector_debug_msg:
                # due to numerical problems this could happen sometimes
                p0 = self._experimental_setup._projections_extrema["p0"][tor_id]
                p1 = self._experimental_setup._projections_extrema["p1"][tor_id]
                print("forward-proj out of bounds",p0, p1)
            return 0
    def BackProjectionSingleTOR(self, img, vec, tor_id):
        """!@brief
            Perform the back-projection of the tor_id-th TOR weighted using vec and save it on the image img 
            @param img: output image 
            @param vec: 
            @param tor_id:
        """
        try:   
            TOR = self.ComputeTOR(tor_id)
            img[TOR["vx"], TOR["vy"], TOR["vz"]] += vec * TOR["prob"]
        except:
            if projector_debug_msg:
                p0 = self._experimental_setup._projections_extrema["p0"][tor_id]
                p1 = self._experimental_setup._projections_extrema["p1"][tor_id]
                print("back-proj out of bounds",p0, p1)

    def ForwardProjection(self, img, TOR_list=None):
        """!@brief
             Implements forward projection of several TORs: 
             If TORList is None forward project all the TORs of the experimental setup 
             If TORList is not None, forward project only those TORs with tor-id in TORList
        """
        # default forward project all the scanner TOR
        if TOR_list is None:
            _TOR_list = range(self.GetNumberOfProjections())
            _Number_Of_TORs = self.GetNumberOfProjections()
        # or if TOR_list!=None forward project only the TOR with ID
        # in the TOR_list
        else:
            _TOR_list = TOR_list
            _Number_Of_TORs = len(TOR_list)
        proj = np.zeros(_Number_Of_TORs, projection_dtype)
        # pdb.set_trace()
        for tor_id, tor in enumerate(_TOR_list):
            proj[tor_id] = self.ForwardProjectSingleTOR(img, tor)
        return proj

    def BackProjection(self, vec, TOR_list=None):
        """!@brief 
            Implements backprojection  
            If TORList is None backproject all the TORs  of the experimental setup 
            If TORList is not None backproject only the TORs with Ids in TORList 
        """

        if TOR_list is None:
            _TOR_list = range(self.GetNumberOfProjections())
            _TOR_Number = self.GetNumberOfProjections()
        # or if TOR_list!=None back project only the TOR with ID
        # in the TOR_list
        else:
            _TOR_list = TOR_list
            _TOR_Number = len(TOR_list)
        #
        img = np.zeros(self.GetNumberOfVoxels(), dtype=voxel_dtype)
        for tor_id, tor in enumerate(_TOR_list):
            # pdb.set_trace()
            if vec[tor_id] == 0:
                continue
            self.BackProjectionSingleTOR(img, vec[tor_id], tor)
        return img

    def Reconstruct(self):
        """!@brief
            Perform niter iterations of the algorithm. The implementation of the update rule is demanded to the derived 
            class. 
        """
        print("Algorithm name {}".format(self._name))
        # this method must be reimplemented in the base class
        self.EvaluateWeightingFactors()
        save_path = 'imgs/01-MLEM'

        for i in range(self._niter):
            start = time.time()
            # this method must be reimplemented in the base class
            self.PerfomSingleIteration()
            # outputfilename = self._output_file_name + "_iter_" + str(i + 1)
            # self.SaveImageToDisk(outputfilename)
            end = time.time()
            print("iteration {0:d} => time: {1:.1f} s".format(i + 1, end - start))

            (self._image/255.0).astype(np.float32).tofile(os.path.join(save_path, str(i) + '.bin'))

        print("Done")
        return self._image

    def Reconstruct_DLMLEM(self):
        """!@brief
            Perform niter iterations of the algorithm. The implementation of the update rule is demanded to the derived 
            class. 
        """
        print("Algorithm name {}".format(self._name))
        # this method must be reimplemented in the base class
        self.EvaluateWeightingFactors()
        self.PerfomMultiIterationDLMLEM()
        return self._image

    def SaveImageToDisk(self, output_file_name):
        """!@brief
            Save the image to file as a np object
        """
        if self._save_img_to_disk:
            Pickle(self._image, output_file_name, ".rec")
