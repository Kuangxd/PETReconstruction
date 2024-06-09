from enum import Enum
import numpy as np


class DetectorType(Enum):
    """!@brief define the detector type
    """
    CT = 1
    PET = 2

class Mode(Enum):
    """!@brief define the source geometry for CT
    """
    PARALLELBEAM = 1
    FANBEAM = 2
    CONEBEAM = 3


class DetectorShape(Enum):
    """!@brief define the pixel geometry for CT
    """
    PLANAR = 1
    ARC = 2
    
class ExperimentalSetup:

    """!@brief
        Implements a generic experimental setup. An experimental setup is made of a field of view, pixels and sources
    """

    def CalculateTRSizeInVoxels(self):
        """!@brief 
        Computes the number of voxels per direction  using the TR size in mm and the voxel size in mm 
        """
        self._voxel_nb = np.rint(
            np.asarray(self.image_matrix_size_mm) / np.asarray(self.voxel_size_mm)
        )

    def GetInfo(self):
        """!@brief
            Return some info about the instance  as a string
        """
        s = ""
        for key, val in self.__dict__.items():
            if (
                key.startswith("_") 
             
            ):
                continue
            s += "{}: {}\n".format(key.strip("_"), val)
        s+='number of projections: {}'.format(self._number_of_projections)
        return s
