import numpy as np

"""
    Dimension of the space 
"""
dims = 3
""" 
    Precision of the floating number used in this project
"""
float_precision_dtype = np.float64
""" 
    This dtype implements a point in 3D physical space
"""
point_dtype = np.dtype(
    [
        ("x", float_precision_dtype),
        ("y", float_precision_dtype),
        ("z", float_precision_dtype),
    ]
)

""" 
    This dtype contains the two physical extrema identifying a projection. 
    The distinction between p0 and p1 is customary. 
"""
proj_extrema_dtype = np.dtype([("p0", point_dtype), ("p1", point_dtype)])


""" 
    
"""
bin_dtype = np.dtype([("s", np.uint), ("theta", np.uint),("slice",np.uint)])

""" 
     This dtype is used to represent a single Tube Of Response  (TOR).
     Each TOR entry contains 3 indices of the voxel followed by the probability 
"""
TOR_dtype = np.dtype(
    [("vx", np.uint), ("vy", np.uint), ("vz", np.uint), ("prob", float_precision_dtype)]
)

"""
    Type of the system matrix element 
"""
voxel_dtype = float_precision_dtype

"""
    Precision of the float in this project
"""
projection_dtype = float_precision_dtype
