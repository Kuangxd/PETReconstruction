import numpy as np
import random
import vtk
from Misc.DataTypes import proj_extrema_dtype, point_dtype
from Geometry.ExperimentalSetup import DetectorType
from Geometry.ExperimentalSetup import ExperimentalSetup
from Misc.Utils import RotatePointAlongZ
from Misc.VisualizationUtils import RenderSceneJupyter, par_text_color
from Misc.VisualizationUtils import (
    CreateAutoOrientedeCube,
    CreateAxesActor,
    CreateTR,
    CreateCornerAnnotation,
    CreateLabel,
    CreateLine
)

from Misc.VisualizationUtils import par_pixel_color, par_background_color
import pdb

default_pixel_size_PET = np.array([1, 1, 10])


class ExperimentalSetupPET(ExperimentalSetup):
    """!@brief Implements an experimental setup containing a PET detector :
        A PET detector consists of a number of sensitive elements (pixels) set in coincidence. 
    """

    def __init__(self):
        self.detector_type = DetectorType.PET

    def GenerateProjectionsExtrema(self):
        """!@brief 
            This method is used to specify those pixels that are set in coincidence: 
            If set a specific half fan size, this means that each pixel can acquire coincidences with  the 2*FanSize+1 in front of it
         """
        single_slice_pairs = []
        # iterate over all the possible combinations of pixels of one detector slice
        for i in range(self.pixels_per_slice_nb):
            for j in range(-self.h_fan_size, self.h_fan_size + 1):
                # this is the id of the pixel facing the current one
                opposite_pixel = (
                    int(i + self.pixels_per_slice_nb / 2.0 + j)
                ) % self.pixels_per_slice_nb
                # this to insert always the pixel with smaller id first
                i0= min(i, opposite_pixel)
                i1 =max(i, opposite_pixel)
                if i0!=i1:
                    single_slice_pairs.append((i0,i1))
                # pdb.set_trace()
        # remove duplicate
        single_slice_pairs = list(set(single_slice_pairs))
        pixel_pairs=[]
        # replicate for all the detector slices 
        for pair  in single_slice_pairs:
            for r1 in range(0,self.detector_slice_nb):
                for r2 in range(0,self.detector_slice_nb):
                    i0=pair[0]+r1*self.pixels_per_slice_nb
                    i1=pair[1]+r2*self.pixels_per_slice_nb
                    pixel_pairs.append((i0,i1))
        # remove duplicate
        pixel_pairs=list(set(pixel_pairs))
        self._number_of_projections = len(pixel_pairs)
        self._projections_extrema = np.zeros(
            self._number_of_projections, dtype=proj_extrema_dtype
        )
        # the line below shuffle the projection extrema. This should improve ART convergence speed. 
        # Try to comment it and run ART and see what happens   
        # random.shuffle(pixel_pairs) 
        # copy the coordinates inside the projections_extrema
        for i,p in enumerate(pixel_pairs):
                    self._projections_extrema["p0"][i] = self._pixel_pos_mm[p[0]]
                    self._projections_extrema["p1"][i] = self._pixel_pos_mm[p[1]]
        self.CalculateTRSizeInVoxels()

    def Update(self):
        """!@brief
            Run all the function to compute the geometry of the PET experimental setup
        """
        if 2 * self.h_fan_size + 1 > self.pixels_per_slice_nb:
            raise Exception("2*h_fan_size +1 should be smaller than number of pixels per slice")
        self._pixel_pos_mm = np.zeros(self.pixels_per_slice_nb, dtype=point_dtype)
        pixel_0 = np.zeros(1, point_dtype)
        # pixel 0  is the one with the largest y coordinate
        pixel_0["x"] = 0
        pixel_0["y"] = self.radius_mm
        pixel_0["z"] = 0
        #
        # pdb.set_trace()
        pixel_angles_deg = np.arange(0, 360.0, 360/self.pixels_per_slice_nb)
        self._pixel_pos_mm = RotatePointAlongZ(pixel_0, pixel_angles_deg)
        self._pixel_pos_mm = np.tile(self._pixel_pos_mm, self.detector_slice_nb)
        # add the z coordinate
        detector_slice_z_coordinates = np.arange(-self.detector_slice_nb + 1, self.detector_slice_nb + 1, 2) / 2 * self.slice_pitch_mm
        # detector slice 0 is the one with the lowest z
        self._pixel_pos_mm["z"]=np.repeat(detector_slice_z_coordinates, self.pixels_per_slice_nb)
        # pdb.set_trace()
        self.GenerateProjectionsExtrema()

    def Draw(self, use_jupyter=0,camera_pos_mm=(0,100,100)):
        """!@brief
             Render an experimental setup containing a PET detector
             @param use_jupyter: if 1 return an image representing the scene, otherwise a vtk interactive rendering is used
             @param camera_pos_mm: triplet representing the position in mm of the camera that is caputring the scene. Note that camera is always pointed towars the TR center 
        """
        ren = vtk.vtkRenderer()
        #  generate cubes representing PET pixels
        for p in self._pixel_pos_mm:
            ren.AddActor(CreateAutoOrientedeCube(p,default_pixel_size_PET,par_pixel_color))
        #  add 3d axis to the scene
        ren.AddActor(CreateAxesActor())
        # add the TR to the scene
        ren.AddActor(CreateTR(self.image_matrix_size_mm, self.voxel_size_mm))
        # add corner annotation
        ren.AddActor(CreateCornerAnnotation((10, 10), self.GetInfo()))
        ren.SetBackground(par_background_color)
        # create the default camera 
        camera =vtk.vtkCamera ()
        camera.SetPosition(camera_pos_mm)
        camera.SetFocalPoint(0, 0, 0)
        ren.SetActiveCamera(camera)
        # if not jupyter use vtk to create a window
        if not use_jupyter:
            renWin = vtk.vtkRenderWindow()
            renWin.AddRenderer(ren)
            # create a render and a window interactor
            iren = vtk.vtkRenderWindowInteractor()
            iren.SetRenderWindow(renWin)
            iren.Start()
        else:
            return RenderSceneJupyter(ren)
