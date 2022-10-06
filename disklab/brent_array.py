import numpy as np
def brent_array(xa,xb,f,tol,aux=None,nitermax=30,delta=None,returninfo=False):
    """
    Brent's root finding algorithm for the equation

       f(x,aux=aux) = 0

    but vectorized, i.e. x is an array and f(x,aux=aux) is the function
    of x. Aux is additional information so that for the different
    array elements different solutions can be found independent of
    each other. Without aux all x values will likely turn out to
    be the same.

    https://en.wikipedia.org/wiki/Brent%27s_method
    
    ARGUMENTS:
      xa, xb     The brackets within which the values of x are to be found.
      f          The function of x to be set to 0. 
      tol        The tolerance in f
      aux        The additional information to distinguish the different x-es.
      nitermax   The maximum nr of iterations allowed

    RETURNS:
      x          The root (array)
      err        The values of f
    
    EXAMPLE SCALAR CASE:

      import numpy as np
      from brent_array import *

      def func(x,aux=None):
         return x**3-1.2

      x,f,info = brent_array(0.,3,func,1e-4,returninfo=True,nitermax=30)

    EXAMPLE ARRAY CASE:

      import numpy as np
      import matplotlib.pyplot as plt
      from brent_array import *
      
      def func(x,aux=None):
          return x**3-aux["rhs"]
      
      n     = 30
      rhs   = np.linspace(0.1,3,n)
      aux   = {"rhs":rhs}
      xa    = np.zeros(n)
      xb    = np.zeros(n)+20.
      x,f,info = brent_array(xa,xb,func,1e-4,returninfo=True,nitermax=30,aux=aux)
      
      plt.plot(rhs,x)
      plt.show()

    """
    #
    # Wenn scalar, dann erst Vector machen
    #
    if np.isscalar(xa): xa = np.array([xa])
    if np.isscalar(xb): xb = np.array([xb])
    xa  = xa*1.0
    xb  = xb*1.0
    tol = tol*1.0
    #
    # Handle delta
    #
    if delta is None:
        delta = xa.copy()*0
    #
    # Calculate the f(xa) and f(xb)
    #
    fa       = f(xa,aux=aux)
    fb       = f(xb,aux=aux)
    #
    # Check if solution bracketed
    #
    assert len(np.where(fa*fb>=0)[0])==0, "Brent: Root not bracketed"
    #
    # Swap where |fb|>|fa|, because xb must be the better guess of the root
    #
    mask     = np.abs(fb)>np.abs(fa)
    fdum     = fa.copy()
    xdum     = xa.copy()
    fa[mask] = fb[mask].copy()
    xa[mask] = xb[mask].copy()
    fb[mask] = fdum[mask]
    xb[mask] = xdum[mask]
    #
    # Set xc, xs and flag
    #
    xc       = xa.copy()
    xd       = xb.copy()
    xs       = xb.copy()
    flag     = xa==xa
    fc       = fa.copy()
    #
    # Condition for iteration
    #
    maskiter = np.abs(fb)>tol
    ii       = np.where(maskiter)[0]
    #
    # If info should be returned
    #
    if returninfo:
        list_xa = []
        list_xb = []
        list_fa = []
        list_fb = []
        list_xa.append(xa.copy())
        list_xb.append(xb.copy())
        list_fa.append(fa.copy())
        list_fb.append(fb.copy())
    #
    # Iteration
    #
    iter     = 0
    while len(ii)>0 and iter<nitermax:
        #
        # Use the inverse quadratic interpolation
        #
        maskk    = np.logical_and(fa!=fc,fb!=fc)
        mask     = np.logical_and(maskk,maskiter)
        xs[mask] = xa[mask]*fb[mask]*fc[mask]/((fa[mask]-fb[mask])*(fa[mask]-fc[mask])) + \
                   xb[mask]*fc[mask]*fa[mask]/((fb[mask]-fc[mask])*(fb[mask]-fa[mask])) + \
                   xc[mask]*fa[mask]*fb[mask]/((fc[mask]-fa[mask])*(fc[mask]-fb[mask]))
        #
        # or else secant method
        #
        maskk    = np.logical_not(maskk)
        mask     = np.logical_and(maskk,maskiter)
        xs[mask] = (xa[mask]*fb[mask]-xb[mask]*fa[mask])/(fb[mask]-fa[mask])
        #
        # Next check if we wish to replace one of
        # these with the bisection method.
        #
        xx       = (3*xa+xb)/4.
        flaginv  = np.logical_not(flag)
        cond1    = (xs-xb)*(xs-xx)>=0
        cond2    = np.logical_and(flag,np.abs(xs-xb)>=np.abs(xb-xc)/2.)
        cond3    = np.logical_and(flaginv,np.abs(xs-xb)>=np.abs(xc-xd)/2.)
        cond4    = np.logical_and(flag,np.abs(xb-xc)<delta)
        cond5    = np.logical_and(flaginv,np.abs(xc-xd)<delta)
        maskk    = np.logical_or(cond1,cond2)
        maskk    = np.logical_or(maskk,cond3)
        maskk    = np.logical_or(maskk,cond4)
        maskk    = np.logical_or(maskk,cond5)
        mask     = np.logical_and(maskk,maskiter)
        xs[mask] = 0.5*(xa+xb)[mask]
        flag[mask] = True
        maskk    = np.logical_not(maskk)
        mask     = np.logical_and(maskk,maskiter)
        flag[mask] = False
        #
        # Compute f(xs)
        #
        fs       = f(xs,aux=aux)
        #
        # Assign new x values
        #
        xd[maskiter] = xc[maskiter].copy()
        xc[maskiter] = xb[maskiter].copy()
        fc[maskiter] = fb[maskiter].copy()
        #
        # Choose which of the two brackets will be xs
        #
        maskk    = fa*fs<0
        mask     = np.logical_and(maskk,maskiter)
        xb[mask] = xs[mask]
        fb[mask] = fs[mask]
        maskk    = np.logical_not(maskk)
        mask     = np.logical_and(maskk,maskiter)
        xa[mask] = xs[mask]
        fa[mask] = fs[mask]
        #
        # Swap where |fb|>|fa|, because xb must be the better guess of the root
        #
        maskk    = np.abs(fb)>np.abs(fa)
        mask     = np.logical_and(maskk,maskiter)
        fdum[mask] = fa[mask]
        xdum[mask] = xa[mask]
        fa[mask] = fb[mask]
        xa[mask] = xb[mask]
        fb[mask] = fdum[mask]
        xb[mask] = xdum[mask]
        #
        # If info should be returned, package these
        #
        if returninfo:
            list_xa.append(xa.copy())
            list_xb.append(xb.copy())
            list_fa.append(fa.copy())
            list_fb.append(fb.copy())
        #
        # Condition for iteration
        #
        maskiter = np.abs(fb)>tol
        ii       = np.where(maskiter)[0]
        iter    += 1
    #
    # Wrap up
    #
    if iter>=nitermax:
        print("No convergence in bisection")
        conv = False
    else:
        conv = True
    if returninfo:
        info = {"iter":iter,"list_xa":list_xa,"list_xb":list_xb,"list_fa":list_fa,"list_fb":list_fb,"converged":conv}
        return xb,fb,info
    else:
        return xb,fb
