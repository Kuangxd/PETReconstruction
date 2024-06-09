import numpy as np
import imageio
import pickle
from Misc.DataTypes import point_dtype
from skimage.measure import block_reduce
import pdb


def FindPolarAngle(v,range_deg=180.0):
    """!@brief
       Evaluate theta 
    """
    return np.rad2deg(np.arctan2(v[0],v[1])) % range_deg

def SafeDivision(A, B):
    """!@brief 
        This function handles the 0 division problem
    """
    eps = 1e-6
    if B == 0:
        return A / (eps)
    return A / B


def CheckParameters(obj,param_name):
    if not hasattr(obj, param_name):
        raise Exception("{} is not defined".format(param_name))
        
    
    
def RotatePointAlongZ(p, angle):
    """!@brief 
    Rotate a rec-array of point_dtype along Z axis
    """
    angle_rad = np.deg2rad(angle)
    shape_0 = max(p.shape[0], angle.shape[0])
    r = np.zeros(shape_0, dtype=point_dtype)
    # pdb.set_trace()
    r["x"] = np.cos(angle_rad) * p["x"] - np.sin(angle_rad) * p["y"]
    r["y"] = np.sin(angle_rad) * p["x"] + np.cos(angle_rad) * p["y"]
    r["z"] = p["z"]
    return r





def ReadImage(filename):
    """!@brief 
     Read a formatted image from disk using imageio module
    """
    img = imageio.imread(filename)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return img





def WriteImage(img, filename):
    """!@brief 
        Write a formatted image from disk using imageio module
    """
    imageio.imwrite(filename, img)




def EvaluatePoint(p, v, size):    
    """!@brief 
    Evaluate the coordinates of a point starting using the formula p+v*size
    @param p:
    @param v:
    @param size: 
     
    """
    return p + v * size





def CheckExt(filename, ext):
    """!@brief 
        Check if a given filename has an extension
    """
    if filename.endswith(ext):
        return filename
    return filename + ext





def DownscaleImage(img,downscale_factors):
    """!@brief 
       Downscale img in each dimension and return the downlscaed image
    """
    return block_reduce(img, downscale_factors, func=np.mean)


def Pickle(obj, filename, ext):
    """!@brief 
        Serialize an object using the pickle library. 
        If the filename does not contain the extension 
        it is added prior to saving file  
    """
    with open(CheckExt(filename, ext), "wb") as f:
        pickle.dump(obj, f)





def Unpickle(filename):
    """!@brief 
        De-serialize an object using the pickle library
    """
    with open(filename, "rb") as f:
        return pickle.load(f)
