import numpy as np
from Algorithms.IterativeReconstruction import IterativeReconstruction
from Misc.DataTypes import projection_dtype, voxel_dtype


class SIRT(IterativeReconstruction):
    """!@brief 
        Implements the Simultaneous Iterative Reconstruction Technique (SIRT) algorithm. P. Gilbert, “Iterative methods for the reconstruction of three dimensional objects from their projections,” 
        J. Theor. Biol., vol. 36, pp. 105-117, 1972. 
    """

    def __init__(self):
        super().__init__()
        self._name = "SIRT"

    def PerfomSingleIteration(self):
        """!@brief
            Implements the update rule for SIRT 
        """
        # forward projection
        proj = self.ForwardProjection(self._image)
        # comparison with experimental measures (subtraction)
        proj -= self._projection_data
        # application normalization factors
        proj *= self._C
        # backprojection and update current estimate
        self._image = self.BackProjection(-proj)* self._R

    # this corresponds to the R matrix of the equation  ???
    # except that instead of storing the matrix we store only
    # the values on the diagonal.

    def __EvaluateRowSum(self):
        self._R = self.BackProjection(
            np.ones(self.GetNumberOfProjections(), dtype=voxel_dtype)
        )
        nnull = self._R != 0
        self._R[nnull] = 1 / self._R[nnull]

    # this corresponds to the C matrix of the equation  ???
    # except that instead of storing the matrix we store only
    # the values on the diagonal

    def __EvaluateColSum(self):
        self._C = np.zeros(self.GetNumberOfProjections(), projection_dtype)
        for i in range(self.GetNumberOfProjections()):
            TOR = self.ComputeTOR(i)
            self._C[i] = np.sum(TOR["prob"])
        nnull = self._C != 0
        self._C[nnull] = 1 / self._C[nnull]

    # compute all the weighting factors needed for the update rule

    def EvaluateWeightingFactors(self):
        self.__EvaluateRowSum()
        self.__EvaluateColSum()
