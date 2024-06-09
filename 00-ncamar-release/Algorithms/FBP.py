import numpy as np
from scipy.interpolate import interp1d
from Misc.DataTypes import float_precision_dtype
from Geometry.ExperimentalSetup import Mode
from Misc.Utils import Pickle
from scipy.interpolate import RegularGridInterpolator 
class FBP:
    """!@brief
         Implements Filtered Back Projection algorithm
    """

    def GenerateRamp(self, tau=1):
        """!@brief
            Generate and return a ramp in the frequency space
            this method is based on .....
        """
        filter_size = self.sinogram._radial_bins
        self._h = np.zeros((filter_size))
        # find all the indices odd with respect to filter_size/2
        idx = np.arange(0, filter_size, 2) 
        f_center=filter_size // 2+ 1-filter_size%2
        self._h[idx] = -(((idx - f_center) * tau * np.pi) ** 2)
        nnull = self._h != 0
        self._h[nnull] = 1 / self._h[nnull]
        self._h[f_center] = 1 / (4 * (tau) ** 2)
        ramp = np.fft.fft(self._h) * self.sinogram._radial_step_mm / 2
        H = abs(ramp)
        # this creates a matrix where each column contains H
        self._Hm = np.array([H] * self.sinogram._angular_bins)
        self._Hm = np.transpose(self._Hm)
        self._Hm = np.tile(self._Hm,self.sinogram._z_bins).reshape(self.sinogram._radial_bins,self.sinogram._angular_bins,-1)
    

    def BackprojectionParalleBeam(self):
        """!@brief: 
            Perform the backprojection for the parallel beam geometry
        """
        # get the TR size in pixel from the sinogram settings
        N = self.sinogram._voxel_nb.astype(int)
        _img = np.zeros(N, float_precision_dtype)
        # x and y are two matrices containing the coordinates of each voxel
        x, y  = np.mgrid[:N[0], :N[0]] -N[0] // 2 
        x = x * self.sinogram.voxel_size_mm[0] + self.sinogram.voxel_size_mm[0] / 2
        y = -(y * self.sinogram.voxel_size_mm[1] + self.sinogram.voxel_size_mm[1] / 2)
        r = np.arange(self.sinogram._radial_bins) - self.sinogram._radial_bins // 2
        r = r * self.sinogram._radial_step_mm + self.sinogram._radial_step_mm / 2
        theta = (
            np.arange(0, self.sinogram._angular_bins)
            * self.sinogram._angular_step_deg
        )
        det_slice_z_coordinates=np.arange(-self.sinogram.detector_slice_nb+1,self.sinogram.detector_slice_nb+1,2)/2*self.sinogram.slice_pitch_mm
        det_slice_z_coordinates.astype(int)

        perc0=perc1=0
        for z_sin,z_det in enumerate(det_slice_z_coordinates):
            perc0=perc1
            perc1=np.trunc((z_sin+1)/len(det_slice_z_coordinates)*100).astype(np.int32)
            if perc1 != perc0:
                print("Backprojecting data, " + str(perc1) + "% done...", end="\r")

            s=self._filterderd_sinogram[:,:,z_sin]
            for sinogram_col, angle in zip(s.T, np.deg2rad(theta)):
                t = y * np.cos(angle) - x * np.sin(angle)
                # interpolate the value of the sinogram
                z_int=int((round(z_det+self.sinogram.image_matrix_size_mm[2]/2)/self.sinogram.voxel_size_mm[2]))
                _img[:,:,z_int] += interp1d(
                    x=r,
                    y=sinogram_col,
                    kind=self.interpolator,
                    bounds_error=False,
                    fill_value=0,
                    assume_sorted=False,
                    )(t)
                
        print("\n")        
        return _img * 2 * np.pi / (self.sinogram._angular_bins)
    
    
    def BackprojectionFanBeam(self):
        """!@brief: 
            Perform the backprojection for the cone beam geometry
        """
        # get the TR size in pixel from the sinogram settings
        N = self.sinogram._voxel_nb.astype(int)
        #
        _img = np.zeros(N, float_precision_dtype)
        # 
        D = self.sinogram.sad_mm
        # calc the magnification
        mag = self.sinogram.sdd_mm / self.sinogram.sad_mm
        # calc the virtual detector pitch
        virt_det_pitch_mm = self.sinogram.detector_pitch_mm/ mag
        #
        u_det = (np.arange(self.sinogram.pixels_per_slice_nb) - self.sinogram.pixels_per_slice_nb // 2 + 0.5) * virt_det_pitch_mm
        # 
        theta = (
            np.arange(0, self.sinogram._angular_bins)
            * self.sinogram._angular_step_deg
        )        # Grid of the coordinates of the pixel centers
        # in the target image (pitch=1 for simplicity)
        x, y  = np.mgrid[:N[0], :N[0]] -N[0] // 2 
        x = x * self.sinogram.voxel_size_mm[0] + self.sinogram.voxel_size_mm[0] / 2
        y = y * self.sinogram.voxel_size_mm[1] + self.sinogram.voxel_size_mm[1] / 2
        r=np.sqrt(np.square(x)+np.square(y))
        varphi=np.arctan2(y,x)
        
        det_slice_z_coordinates=np.arange(-self.sinogram.detector_slice_nb+1,self.sinogram.detector_slice_nb+1,2)/2*self.sinogram.slice_pitch_mm
        det_slice_z_coordinates.astype(int)
        # loop over sinogram slices

        perc0=perc1=0
        for z_sin,z_det in enumerate(det_slice_z_coordinates):
            perc0=perc1
            perc1=np.trunc((z_sin+1)/len(det_slice_z_coordinates)*100).astype(np.int32)
            if perc1 != perc0:
                print("Backprojecting data, " + str(perc1) + "% done...", end="\r")
            # get current slice
            s=self._filterderd_sinogram[:,:,z_sin]
            
            for (filtered_proj, curr_beta_rad) in zip(s.T, np.deg2rad(theta)):
                
                z_int=int((round(z_det+self.sinogram.image_matrix_size_mm[2]/2)/self.sinogram.voxel_size_mm[2]))
                U = D + r * np.sin(varphi-curr_beta_rad)
                u = r * np.cos(varphi-curr_beta_rad) * D / U
                W_BP = np.square(U)    
                #print (sigma_det)
                _img[:,:,z_int] += interp1d(
                    x = u_det,
                    y = filtered_proj,
                    kind =self.interpolator,
                    bounds_error = False,
                    fill_value = 0,
                    assume_sorted = False
                    ) (u) / W_BP
        print("\n")
        return _img 

    def BackprojectionConeBeam(self):
        """!@brief: 
            Perform the backprojection for the cone beam geometry
        """
        # By these three variables, we can control
        # the angular range and step in backprojection
        
        N = self.sinogram._voxel_nb.astype(int)
        _img = np.zeros(N, float_precision_dtype)
        D = self.sinogram.sad_mm
        x, y ,z  = np.mgrid[:N[0], :N[0],:N[2]]
        x-=N[0]//2
        y-=N[0]//2
        z-=N[2]//2 
        x = (x+0.5) * self.sinogram.voxel_size_mm[0]
        y = -(y+0.5) * self.sinogram.voxel_size_mm[1]
        z = (z+0.5) * self.sinogram.voxel_size_mm[2]
        r=np.sqrt(np.square(x)+np.square(y))
        varphi=np.arctan2(-y,x)
        theta = (
            np.arange(0, self.sinogram._angular_bins)
            * self.sinogram._angular_step_deg
        )      
        # calc the magnification
        mag = self.sinogram.sdd_mm / self.sinogram.sad_mm
        # calc the virtual detector pitch
        virt_det_pitch_mm = self.sinogram.detector_pitch_mm / mag
        # Create an array with the radial and longitudinal coordinates
        # of the 2D detector pixels
        
        u_det,v_det = np.mgrid[-self.sinogram.pixels_per_slice_nb // 2 + 0.5 :  self.sinogram.pixels_per_slice_nb // 2 + 0.5 : 1,
                                     -self.sinogram.detector_slice_nb   // 2 + 0.5 :  self.sinogram.detector_slice_nb   // 2 + 0.5 : 1
                                     ] * virt_det_pitch_mm
        

        
        
        rearranged_proj=np.transpose(self._filterderd_sinogram,(1,0,2))
        u_det_1d = u_det[:,0]
        v_det_1d = v_det[0,:]
        #print ('filt sino',rearranged_proj.shape)
        #print ('u ',u_det_1d.shape)
        #print ('v ',v_det_1d.shape)
       
        # Iterate over all the selected projections    
        perc0=perc1=0
        iproj = 0
        for (filtered_proj, beta_rad) in zip(rearranged_proj, np.deg2rad(theta)):
            iproj += 1
            perc0=perc1
            perc1=np.trunc(iproj/len(np.deg2rad(theta))*100).astype(np.int32)
            if perc1 != perc0:
                print("Backprojecting data, " + str(perc1) + "% done...", end="\r")

            U = D + r * np.sin(varphi-beta_rad)
            u = r * np.cos(varphi-beta_rad) * D / U
            v = z * D / U
            W_BP = np.square(U)            
            #print ('filt proj',filtered_proj.shape)
            # Accumulate the weighted values in the destination image
            _img += RegularGridInterpolator(
                points = (u_det_1d, v_det_1d),
                values = filtered_proj,
                method = self.interpolator,
                bounds_error = False, 
                fill_value = 0
            )  ((u,v)) / W_BP

    
        return _img
        
    
    def Backprojection(self):
        """!@brief
            Perform backprojection 
        """
        if self.sinogram.mode==Mode.PARALLELBEAM:
            return self.BackprojectionParalleBeam()
        elif self.sinogram.mode==Mode.CONEBEAM:
            return self.BackprojectionConeBeam()
        elif self.sinogram.mode==Mode.FANBEAM:
            return self.BackprojectionFanBeam()
    def Reconstruct(self):
        """!@brief 
             Run the FBP reconstruction and return the reconstructed image
        """
        # generate the ramp filter for filtering the sino
        print("Generating ramp filter... ")
        self.GenerateRamp()
        print("done.\n")
        # perform FFT of the sinogram along R
        print("Filtering sinogram data... ")
        self.fft1d_sinogram = np.fft.fft(self.sinogram._data, axis=0)
        # filter and perform 2D
        self._filterderd_sinogram = np.real(
            np.fft.ifft(self.fft1d_sinogram * self._Hm, axis=0)
        )
        print("done.\n")
        self._image = self.Backprojection()
        print("Reconstruction done.\n")

    def SaveImageToDisk(self, output_file_name):
        """!@brief
            Save the reconstructed image to file
        """
        Pickle(self._image, output_file_name, ".rec")
