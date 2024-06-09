import numpy as np
from Algorithms.IterativeReconstruction import IterativeReconstruction
import pdb
import cv2
import matplotlib
import math
import os
from numpy.linalg import svd
import random

from dl_function import *

class MLEM(IterativeReconstruction):
    """!@brief  
    Implements the Maxmimum Likelihood Estimation Maximization algorithm (MLEM).  
    L. A. Shepp and Y. Vardi, "Maximum Likelihood Reconstruction for Emission Tomography,"
    in IEEE Transactions on Medical Imaging, vol. 1, no. 2, pp. 113-122, Oct. 1982.
    """

    def __init__(self):
        super().__init__()
        self._name = "MLEM"

    def PerfomSingleIteration(self):
        """!@brief 
            Implements the update rule for MLEM
        """
        # forward projection
        proj = self.ForwardProjection(self._image)
        # this avoid 0 division
        nnull = proj != 0
        # comparison with experimental measures (ratio)
        proj[nnull] = self._projection_data[nnull] / proj[nnull]
        #  backprojection
        tmp = self.BackProjection(proj)
        # apply sensitivity correction and update current estimate 
        self._image = self._image * self._S * tmp
        # pdb.set_trace()

    def img2uint8(self, img):
        img[img < 0] = 0
        img[img > 255] = 255
        return img.astype(np.uint8)
    
    
    def PerfomMultiIterationDLMLEM(self):
        """!@brief 
            Implements the update rule for MLEM
        """
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)

        #######################################################################################
        try:
            u = np.fromfile(self.args.network_init_img_path, np.float64).reshape(128, 128, 4) * 255.0
        except:
            u = np.fromfile(self.args.network_init_img_path, np.float32).reshape(128, 128, 4) * 255.0


        x = dl_function(u)      

        if self.args.w_svd == 0:
            u = x


        v = np.zeros((128, 128, 1))

        p = 1 / self._S
        c = 0.5 #0.5
        # pdb.set_trace()

        cv2.imwrite(self.args.save_path+'only-use-network.jpg', 255-self.img2uint8(x[:, :, -1]))
        (x[:, :, -1]/255.0).astype(np.float32).tofile(self.args.save_path+'only-use-network.bin')
        model_pred_img = x[:, :, -1]
        print('p.max():%.4f, p.min():%.4f' % (p.max(), p.min()))

        for j in range(100):
            fu = dl_function(u)

            for i in range(1):
                # forward projection
                proj = self.ForwardProjection(x)
                # this avoid 0 division
                nnull = proj != 0
                # comparison with experimental measures (ratio)
                proj[nnull] = self._projection_data[nnull] / proj[nnull]
                #  backprojection
                tmp = self.BackProjection(proj)
                # apply sensitivity correction and update current estimate 
                Q = x / p * tmp

                # 1
                uvh = np.zeros((128, 128, 4))
                for ii in range(4):
                    uu, ss, vh = np.linalg.svd(x[:,:,ii])
                    temp_w = ss * self.args.w_svd
                    uvh[:, :, ii] = np.dot(uu*(temp_w), vh)

                if self.args.w_svd:
                    temp_v = fu - (p + v - uvh) / c
                else:
                    temp_v = fu - (p + v) / c


                x = 0.5 * (temp_v + np.sqrt(np.square(temp_v) + 4 * p * Q / c))

                cv2.imwrite(self.args.save_path + str(j) + '-' + str(i) + '-x.jpg', x[:, :, -1].astype(np.uint8))

            v_momentum = 0
            for i in range(5):
                grad = dl_function_grad(u, x, v/c)
                u_grad = grad
                v_momentum = 0.9 * v_momentum + 0.1 * u_grad
                u = u - v_momentum * 10

                
                print('grad_max:%.2f grad_min:%.2f umax:%.2f umin:%.2f xmax:%.2f xmin:%.2f v/cmax:%.2f v/cmin:%.2f' % 
                        (u_grad.max(), u_grad.min(), u.max(), u.min(), x.max(), x.min(), (v/c).max(), (v/c).min()))
                u[u < 0] = 0.0   

            fu = dl_function(u)            
            v = v + c * (x - fu)
            # pdb.set_trace()

            # 保存结果
            cv2.imwrite(self.args.save_path+str(j)+'_x-v-c.jpg', 255-self.img2uint8((x+v/c)[:,:,-1]))
            cv2.imwrite(self.args.save_path+str(j)+'_u.jpg', 255-self.img2uint8(u[:,:,-1]))
            cv2.imwrite(self.args.save_path+str(j)+'_fu.jpg', 255-self.img2uint8(fu[:,:,-1]))
            cv2.imwrite(self.args.save_path+'000_lastest_fu.jpg', 255-self.img2uint8(fu[:,:,-1]))
            (fu[:,:,-1]/255.0).astype(np.float32).tofile(self.args.save_path+str(j)+'_fu.bin')
            (u/255.0).astype(np.float32).tofile(self.args.save_path+str(j)+'_u.bin')
            (fu[:,:,-1]/255.0).astype(np.float32).tofile(self.args.save_path+'000_lastest_fu.bin')

    def __EvaluateSensitivity(self):
        """!@brief
             Backproject a vector filled with 1: the obtained image is often called
            sensitivity image
        """
        self._S = self.BackProjection(np.ones(self.GetNumberOfProjections()))
        nnull = self._S != 0
        self._S[nnull] = 1 / self._S[nnull]

    def EvaluateWeightingFactors(self):
        """!@brief
            Compute all the weighting factors needed for the update rule
        """
        self.__EvaluateSensitivity()
