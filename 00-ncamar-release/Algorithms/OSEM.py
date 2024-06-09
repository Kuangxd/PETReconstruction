import numpy as np
from Algorithms.IterativeReconstruction import IterativeReconstruction
from Misc.DataTypes import projection_dtype


class OSEM(IterativeReconstruction):
    """!@brief
    Implements the Order Subset Estimation Maximization (OSEM)  algorithm described in the paper Hudson, H.M., Larkin, R.S. (1994) 
    "Accelerated image reconstruction using ordered subsets of projection data", IEEE Trans. Medical Imaging, 13 (4), 601–609
    """

    def __init__(self):
        super().__init__()
        self._name = "OSEM"

    def PerfomSingleSubsetIteration(self, subset):
        """
        @!brief 
            Implements the update rule for OSEM formula 
        """
        # Get all the projections belonging to the current subset
        LORCurrentSubset = self._subsets[subset]
        # perform the forward projection using only projection 
        # belonging to the current subset 
        proj = self.ForwardProjection(self._image, LORCurrentSubset)
        # avoid 0 division
        inv_proj = np.zeros_like(proj, projection_dtype)
        nnull = proj != 0
        # comparison with experimental measures of the current subset (ratio)
        inv_proj[nnull] = self._data_subsets[subset][nnull] / proj[nnull]
        # perform the forward projection using only projection 
        # belonging to the current subset
        tmp = self.BackProjection(inv_proj, LORCurrentSubset)
        # apply sensitivity correction for current subset and update current estimate 
        self._image = self._image * self._S[subset] * tmp

    def PerfomSingleIteration(self):
        """!@brief
            Implements the update rule for OSEM. One loop over all the subsets is considered one OSEM iteration
        """

        for subset in range(self._subsetnumber):
            self.PerfomSingleSubsetIteration(subset)

    def __EvaluateSensitivity(self):
        """!@brief 
            Calculate the sensitivity of each subset
        """
        self._S = []
        # calculate the sensitivity of each subset
        for subset in range(self._subsetnumber):
            LORCurrentSubset = self._subsets[subset]
            NumberOfLORCurrentSubset = len(LORCurrentSubset)
            tmp = self.BackProjection(
                np.ones(NumberOfLORCurrentSubset), LORCurrentSubset
            )
            # store inverted sensitivity
            nnull = tmp != 0
            tmp[nnull] = 1 / tmp[nnull]
            self._S.append(tmp)

    def EvaluateWeightingFactors(self):
        """!@brief
             Compute all the weighting factors needed for the update rule
        """
        self.__CreateSubsets()
        self.__EvaluateSensitivity()

    def __CreateSubsets(self):
        """!@brief
             Shuffle and divide data into random subsets
        """
        # get the indices of all the LOR of the system
        LORIds = np.array(range(self.GetNumberOfProjections()))
        # shuffle the LOR indices
        np.random.shuffle(LORIds)
        # divide the indices into subset
        self._subsets = np.split(LORIds, self._subsetnumber)
        # associate each _projection_data with a subset
        self._data_subsets = []
        for current_subset in range(self._subsetnumber):
            self._data_subsets.append(
                self._projection_data[self._subsets[current_subset]]
            )
