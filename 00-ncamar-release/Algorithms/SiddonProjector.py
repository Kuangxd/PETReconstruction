import numpy as np
import math
from Misc.DataTypes import TOR_dtype
from Misc.Utils import SafeDivision
import pdb

class SiddonProjector:
    """!@brief 
    Implements the Siddon's algorithm. The algorithm details can be found in Siddon, R. L. (1985), 
    Fast calculation of the exact radiological path for a three‐dimensional CT array. Med. Phys., 12: 252-255. doi:10.1118/1.595715 
    """

    def __init__(self, image_matrix_size_mm, voxel_size_mm):
        """!@brief
            Initialize all the variables common to each ray-tracing procedure
        """
        # Number of voxels
        self.image_matrix_size_mm = np.array(image_matrix_size_mm)
        self.voxel_size_mm = np.array(voxel_size_mm)
        self._voxel_nb = np.rint(
            self.image_matrix_size_mm / self.voxel_size_mm
        ).astype(np.int_)
        # pdb.set_trace()
        # Coordinates of the planes representing the grid
        # equation 3 of the paper
        # For sake of simplicity the center of the TR center is assumed to be
        # in the coordinate (0,0,0)
        self._Xplanes = np.array(
            [
                -self.image_matrix_size_mm[0] / 2 + i * self.voxel_size_mm[0]
                for i in range(self._voxel_nb[0] + 1)
            ]
        )
        self._Yplanes = np.array(
            [
                -self.image_matrix_size_mm[1] / 2 + i * self.voxel_size_mm[1]
                for i in range(self._voxel_nb[1] + 1)
            ]
        )
        self._Zplanes = np.array(
            [
                -self.image_matrix_size_mm[2] / 2 + i * self.voxel_size_mm[2]
                for i in range(self._voxel_nb[2] + 1)
            ]
        )
    def CalcIntersection(self, P1, P2):
        """!@brief
            Calculate the path according to the Siddon's algorithm.
            Trace the line connecting P1 to P2  and return an np array containing the 
            indices of the intersected voxels and the intersection length 
            @param P1: point in physical space to start ray-tracing
            @param P2: point in physical space to end ray-tracing
        """
        # pdb.set_trace()
        # find the parametric intersection with all the sets of planes
        # equation 4 of the paper
        alphaX = SafeDivision(self._Xplanes - P1["x"], P2["x"] - P1["x"])
        alphaY = SafeDivision(self._Yplanes - P1["y"], P2["y"] - P1["y"])
        alphaZ = SafeDivision(self._Zplanes - P1["z"], P2["z"] - P1["z"])
        # alphaX[0]  represents the first intersection with plane parallel to X
        # alphaX[-1] represents the last intersection with plane parallel to X etc
        # same for alphaY and alphaZ
        # equation 5 of the paper

        alphaMin = max(
                    0,
                    min(alphaX[0], alphaX[-1]),
                    min(alphaY[0], alphaY[-1]),
                    min(alphaZ[0], alphaZ[-1])
                    )
    
        alphaMax = min( 
                    1,
                    max(alphaX[0], alphaX[-1]),
                    max(alphaY[0], alphaY[-1]),
                    max(alphaZ[0], alphaZ[-1])
                    )
        if alphaMax <= alphaMin:
            # the ray does not intersect the TR
            # return an empty TOR
            return np.empty((0), dtype=TOR_dtype)
        # merge the sets alphaX,alphaY,alphaZ
        # equation 8 of the paper
        alpha = np.array(
            np.concatenate(([alphaMin], alphaX, alphaY, alphaZ, [alphaMax]))
        )
        # cut alpha values smaller than alphaMin and bigger than alphaMax
        alpha = alpha[np.logical_and(alpha >= alphaMin, alpha <= alphaMax)]
        # remove duplicates from the alpha vector        
        alpha = np.unique(alpha)
        # sort alpha values
        alpha.sort()
        # calculate the Euclidean distance from P1 to P2
        # equation 11 of the paper
        d12 = math.sqrt(
              (P1["x"] - P2["x"]) ** 2
            + (P1["y"] - P2["y"]) ** 2
            + (P1["z"] - P2["z"]) ** 2
        )
        # equation 13 of the paper
        alphamid = (alpha[:-1] + alpha[1:]) / 2.0
        # allocate the np array containing the intersected voxels
        # and the relative intersection length
        TOR = np.zeros(alphamid.shape, dtype=TOR_dtype)
        # equation 10 of the paper
        TOR["prob"] = d12 * (alpha[1:] - alpha[:-1])
        # equation 12 of the paper
        TOR["vx"] = (
            (P1["x"] + alphamid * (P2["x"] - P1["x"]) - self._Xplanes[0])
            // self.voxel_size_mm[0]
        ).astype(int)
        TOR["vy"] = (
            (P1["y"] + alphamid * (P2["y"] - P1["y"]) - self._Yplanes[0])
            // self.voxel_size_mm[1]
        ).astype(int)
        TOR["vz"] = (
            (P1["z"] + alphamid * (P2["z"] - P1["z"]) - self._Zplanes[0])
            // self.voxel_size_mm[2]
        ).astype(int)
        return TOR

    def DrawLine(self, P1, P2):
        """@!brief 
            DrawLine the result of the ray-tracing in a 3d image
            @param P1: point in physical space to start ray-tracing
            @param P2: point in physical space to end ray-tracing
        """
        res = self.CalcIntersection(P1, P2)
        img = np.zeros((self._voxel_nb))
        if len(res) == 0:
            return img
        img[res["vx"], res["vy"], res["vz"]] = res["prob"]
        return img
