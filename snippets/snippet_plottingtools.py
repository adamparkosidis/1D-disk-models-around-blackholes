import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def surface(f,x=None,y=None,rstride=None,cstride=None,
            xlabel=None,ylabel=None,zlabel=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    nx = f.shape[0]
    ny = f.shape[1]
    if x is None:
        x = np.linspace(0,nx-1,nx)
    if y is None:
        y = np.linspace(0,ny-1,ny)
    if rstride is None:
        rstride = 1
    if cstride is None:
        cstride = 1
    xx, yy = np.meshgrid(x, y,indexing='ij')
    ax.plot_wireframe(xx, yy, f, rstride=rstride, cstride=cstride)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if zlabel is not None: ax.set_zlabel(zlabel)
    return fig,ax
