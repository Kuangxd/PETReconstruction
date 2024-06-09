import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pdb

def Visualize3dImage(img, slice_axis=2, _cmap='gray'):
    """!@brief display a 3d ndarray with a slider to move along the third dimension. 
         Extra keyword arguments are passed to imshow
         @param img
         @param slice_axis
         @param params 
    """
    # check dim
    if not img.ndim == 3:
        raise ValueError("img should be an ndarray with ndim == 3")
    transpose_rule=[0,1,2]
    transpose_rule[2],transpose_rule[slice_axis]=transpose_rule[slice_axis],transpose_rule[2]
    t_img=np.transpose(img, tuple(transpose_rule))
    # create figure
    fig=plt.figure()
    ax1= plt.Axes(fig, [0.1, 0.1, 0.6, .80])
    fig.add_axes(ax1)
    current_slice = t_img[:,:,0]

    # display image
    l = ax1.imshow(current_slice,cmap=_cmap)
    # define slider
    ax = fig.add_axes([0.1, 0, 0.6, 0.03])
    # create the slider
    slice_max=img.shape[slice_axis] - 1
    slice_min=0
    slider = Slider(ax, 'slice', slice_min, slice_max,valinit=0, valfmt='%i')    
    def UpdateImage(event):
        slice_number = int(slider.val)
        current_slice = t_img[:,:,slice_number]
        l.set_clim(np.min(current_slice),np.max(current_slice))
        l.set_data(current_slice)
        fig.canvas.draw_idle()
    def Move(event):
        if event.key=='right':
            val=min(slice_max,slider.val+1)
            slider.set_val(val)
        if event.key=='left':
            val=max(0,slider.val-1)
            slider.set_val(val)
        UpdateImage(event)
    fig.canvas.mpl_connect('key_press_event', Move)
    slider.on_changed(UpdateImage)
