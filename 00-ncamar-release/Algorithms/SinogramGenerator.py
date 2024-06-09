import numpy as np
from Algorithms.SiddonProjector import SiddonProjector
from Misc.Sinogram import Sinogram
from Geometry.ExperimentalSetup import Mode
import pdb

class SinogramGenerator:
    """!@brief
       This class  generates a Sinogram of an image using the projection_extrema contained in the detector
    """

    def __init__(self, experimental_setup):
        self._experimental_setup = experimental_setup

    def GenerateObjectSinogram(self, img, transponse_image=0):
        """!@brief 
            Generate the Sinogram of a 2D image. The binning parameters are taken from the experimental setup  
        """
        # copy some of the parameters inside the sinogram
        self.sinogram = Sinogram(self._experimental_setup.pixels_per_slice_nb,
                                  self._experimental_setup.gantry_angles_nb,
                                  self._experimental_setup.detector_slice_nb)
        _img = img
        if transponse_image == 1:
            _img = np.transpose(img, axes=(1, 0, 2))
        else:
            _img = img
        Siddon = SiddonProjector(
            self._experimental_setup.image_matrix_size_mm,
            self._experimental_setup.voxel_size_mm,
        )
        StartPoints = self._experimental_setup._projections_extrema["p0"]
        EndPoints =   self._experimental_setup._projections_extrema["p1"]
        # for convenience we copy the usel attributes of the geometry inside the sinogram object
        self.sinogram.image_matrix_size_mm = self._experimental_setup.image_matrix_size_mm
        self.sinogram.voxel_size_mm = self._experimental_setup.voxel_size_mm
        self.sinogram._radial_step_mm=self._experimental_setup._radial_step_mm
        self.sinogram.mode=self._experimental_setup.mode
        self.sinogram._voxel_nb=self._experimental_setup._voxel_nb
        self.sinogram._angular_step_deg=self._experimental_setup._angular_step_deg
        self.sinogram.gantry_angles_nb=self._experimental_setup.gantry_angles_nb
        self.sinogram.detector_slice_nb=self._experimental_setup.detector_slice_nb
        self.sinogram.slice_pitch_mm=self._experimental_setup.slice_pitch_mm
#        print(self.sinogram.geometry)
        if self._experimental_setup.mode != Mode.PARALLELBEAM:
            self.sinogram.sad_mm= self._experimental_setup.sad_mm
            self.sinogram.sdd_mm= self._experimental_setup.sdd_mm
            
        self.sinogram.detector_pitch_mm= self._experimental_setup._detector_pitch_mm
        self.sinogram.pixels_per_slice_nb=self._experimental_setup.pixels_per_slice_nb
        perc0=perc1=0
        for i in range(len(StartPoints)):
            perc0=perc1
            perc1=np.trunc((i+1)/len(StartPoints)*100).astype(np.int32)
            if perc1 != perc0:
                print("Projecting data, " + str(perc1) + "% done...", end="\r")
            # pdb.set_trace()
            TOR = Siddon.CalcIntersection(StartPoints[i], EndPoints[i])
            proj = np.sum(_img[TOR["vx"], TOR["vy"], TOR["vz"]] * TOR["prob"])
            self.sinogram.AddItem(self._experimental_setup._bins['s'][i], self._experimental_setup._bins[i]['theta'], self._experimental_setup._bins[i]['slice'], proj)
        return self.sinogram
