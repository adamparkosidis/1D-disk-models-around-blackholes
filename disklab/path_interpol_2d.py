import numpy as np
import scipy.ndimage
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt

def path_interpol_2d(z,x,y,xpath,ypath,method='linear',use_interp2d=False):
    """
    Given is a 2-D array of values z[0:nx,0:ny] on a rectangular grid
    given by the 1-D arrays x[0:nx] and y[0:ny] for the dimensions
    nx>0 and ny>0. 
    Given are two 1-D arrays xpath[0:nlen], ypath[0:nlen], which are
    the coordinates of points along a path in the 2-D space. 
    Returned will be a 1-D array of values zpath[0:nlen] which are
    the interpolated values of the 2-D array z[:,:] at the points
    along the path. 

    Note: For a single value of xpath,ypath (a point instead of a
          path) it is more useful to use scipy.interpolate.interp2d(). The
          main point of path_interpol_2d is to be faster than
          repetitive calls to scipy.interpoalte.interp2d(). But by setting
          use_interp2d=True you can force the use of interp2d
          also for the path (i.e. repetitive call to interp2d).

    Note: Inspiration from https://stackoverflow.com/questions/7878398/how-to-extract-an-arbitrary-line-of-values-from-a-numpy-array

    ARGUMENTS:
      z            The 2-D array to be interpolated
      x            The 1-D x coordinate of the 2-D array z(x,y)
      y            The 1-D y coordinate of the 2-D array z(x,y)
      xpath        The x-coordinates of points along the path (1-D array)
      ypath        The x-coordinates of points along the path (1-D array)
      method       Can either be 'linear', 'cubic' or (when using interp2d) 'quintic'
      use_interp2d If True, then do interpolation with scipy.interpolate.interp2d
    
    EXAMPLE:
      from path_interpol_2d import *
      x, y  = np.mgrid[-5:5:0.1, -2:8:0.1]
      z     = np.sqrt(x**2 + y**2) + np.sin(x**2 + y**2)
      npath = 100
      xpath = x.min() + 0.5*(x.max()-x.min())*np.linspace(0.,1.,npath)
      ypath = y.min() + (y.max()-y.min())*np.linspace(0.,1.,npath)
      zpath = path_interpol_2d(z,x[:,0],y[0,:],xpath,ypath)
      zpathalt = path_interpol_2d(z,x[:,0],y[0,:],xpath,ypath,use_interp2d=True)
      fig, axes = plt.subplots(nrows=2)
      extent = (x.min(), x.max(), y.min(), y.max())
      axes[0].imshow(z,extent=extent)
      axes[0].plot(xpath, ypath)
      axes[0].axis('image')
      axes[1].plot(zpath,label='ndimage')
      axes[1].plot(zpathalt,'o',label='interp2d')
      axes[1].legend(loc='lower right')
      plt.show()

    """
    assert len(x.shape)==1, 'x should be 1-D array'
    assert len(y.shape)==1, 'y should be 1-D array'
    assert len(z.shape)==2, 'z should be 1-D array'
    assert len(x)==z.shape[0], 'x length should equal length of z[:,0]'
    assert len(y)==z.shape[1], 'x length should equal length of z[0,:]'
    assert len(xpath)==len(ypath), 'xpath and ypath must have same length'
    #
    # Choose method
    #
    if use_interp2d:
        #
        # Use the slower method, but using standard scipy.interp2d
        #
        npath = len(xpath)
        f     = interp2d(x, y, z, kind=method)
        zi    = np.zeros(npath)
        for ip in range(npath):
            zi[ip] = f(xpath[ip],ypath[ip])
    else:
        #
        # Use the faster method using ndimage
        #
        # First interpolate the xpath and ypath as index coordinates
        # onto x and y
        #
        nx = len(x)
        ny = len(y)
        ix = np.interp(xpath,x,np.linspace(0,nx-1,nx))
        iy = np.interp(ypath,y,np.linspace(0,ny-1,ny))
        #
        # Now use ndimage to interpolate these values 
        #
        if method=='linear':
            order = 1
        elif method=='cubic':
            order = 3
        else:
            raise ValueError('Method for path_interpol_2d() not valid.')
        zi = scipy.ndimage.map_coordinates(z, np.vstack((iy,ix)), order=order)
    #
    # Return the values
    #
    return zi
