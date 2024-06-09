import numpy as np
from Algorithms.IterativeReconstruction import IterativeReconstruction
from Misc.DataTypes import projection_dtype


class ART(IterativeReconstruction):
    """!@brief 
    Implements the Algebraic Reconstruction Technique (ART) Algorithm. Richard Gordon, Robert Bender, Gabor T. Herman,
    Algebraic Reconstruction Techniques (ART) for three-dimensional electron microscopy and X-ray photography,
    Journal of Theoretical Biology, Volume 29, Issue 3, 1970, Pages 471-481, ISSN 0022-5193, """

    def __init__(self):
        super().__init__()
        self._name = "ART"

    def PerfomSingleIteration(self):
        """!@brief 
            Implements the update rule for ART. Note that in the ART algorithm the image is updated after each projection is processed. 
            An ART iteration is defined when the algorithm has gone   through all the projections  
        """
        for l in range(self.GetNumberOfProjections()):
            # forward projection         
            proj = self.ForwardProjectSingleTOR(self._image, l)
            # comparison with expertimental measures (subtraction)
            proj -= self._projection_data[l]
            # application normalization factors
            proj *= self._L[l]
            # backprojection and update current estimate
            self.BackProjectionSingleTOR(self._image, -proj, l)

    def __EvalNormalization(self):
        """!@brief
            Evaluate the |A|^2 normalization factor
        """
        self._L = np.zeros(
            self.GetNumberOfProjections(), dtype=projection_dtype
        )
        for i in range(self.GetNumberOfProjections()):
            TOR = self.ComputeTOR(i)
            self._L[i] = np.sum(np.square(TOR["prob"]))
        nnull = self._L != 0
        self._L[nnull] = 1 / self._L[nnull]

    def EvaluateWeightingFactors(self):
        """!@brief
            Compute all the weighting factors needed during the update rule
        """
        self.__EvalNormalization()
