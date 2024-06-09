import numpy as np
from Misc.DataTypes import  float_precision_dtype

class Sinogram:
    """!@brief
        Implements sinogram
    """

    def __init__(self, r_bins, theta_bins, z_bins):
        """!@brief
            Calculate the number of angular and radial bins and allocate the sinogram matrix 
        """
        self._angular_bins = theta_bins
        self._radial_bins =  r_bins
        self._z_bins=        z_bins
        self._data = np.zeros(
            (self._radial_bins, self._angular_bins, self._z_bins), dtype=float_precision_dtype
        )
    def AddItem(self, r_id, theta_id, z_id, val):
        self._data[r_id, theta_id, z_id] += val


