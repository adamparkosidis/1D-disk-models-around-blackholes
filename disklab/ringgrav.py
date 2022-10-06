import numpy as np
from scipy.interpolate import interp1d

class ringgrav(object):
    """
    A class to compute and tabulate the gravitational force (actually:
    acceleration) of a pointlike test particle in the plane of a gravitating
    ring. 

    ARGUMENTS:
     rring      = Radius of the ring
     gmring     = G times the mass of the ring
     nang       = Nr of angular points used in the computation. With nang=100 
                  you get typically 1e-3 relative accuracy.
     rsmooth    = (if not None) Gravitational smoothing radius
    """

    def __init__(self,rring=1.0,gmring=1.0,nang=100,rsmooth=None):
        self.rring   = rring
        self.gmring  = gmring
        self.nang    = nang
        self.rsmooth = rsmooth

    def ringgrav(self,rp):
        """
        Returns the gravitational acceleration of a pointlike test particle
        in the plane of the ring, when the particle is at radius rp. 
        """
        x0     = rp
        ang    = 0.5 * np.pi * np.linspace(0.,1.-1./(2*self.nang-1.),self.nang)
        slope  = np.tan(ang)
        slope2 = slope**2
        a      = slope2+1.
        b      = -2*slope2*x0
        c      = slope2*x0**2-self.rring**2
        det    = b**2-4*a*c
        xp     = np.zeros(self.nang)
        xn     = np.zeros(self.nang)
        yp     = np.zeros(self.nang)
        yn     = np.zeros(self.nang)
        sqdet  = np.zeros(self.nang)
        ipos   = np.where(det>0.)
        sqdet[ipos] = np.sqrt(det[ipos])
        xp[ipos]    = (-b[ipos]+sqdet[ipos])/(2*a[ipos])
        xn[ipos]    = (-b[ipos]-sqdet[ipos])/(2*a[ipos])
        yp[ipos]    = np.abs(slope[ipos]*(xp[ipos]-x0))
        yn[ipos]    = np.abs(slope[ipos]*(xn[ipos]-x0))
        xp     = xp[ipos]
        xn     = xn[ipos]
        yp     = yp[ipos]
        yn     = yn[ipos]
        x      = np.hstack((xp,xn[::-1]))
        y      = np.hstack((yp,yn[::-1]))
        phi    = np.arctan(y/x)
        phi[phi<0] = phi[phi<0] + np.pi
        phi[-1] = np.pi
        xav    = 0.5 * ( x[1:] + x[:-1] )
        yav    = 0.5 * ( y[1:] + y[:-1] )
        dphi   = phi[1:] - phi[:-1]
        dgm    = 2.0 * self.gmring * dphi / (2.0*np.pi)   # Factor of 2: Upper and lower half
        dist   = np.sqrt((xav-x0)**2+yav**2)
        if self.rsmooth is None:
            disti3 = 1.0/dist**3
        else:
            disti3 = 1.0/((dist**2+self.rsmooth**2)*dist)
        fx     = disti3*dgm*(xav-x0)
        fxtot  = fx.sum()
        return fxtot

    def approx_ringgrav_zypman(self,rp):
        """
        As ringgrav() but now according to the analytic approximation
        of Fredy R. Zypman (2006) Americal Journal of Physics 74, 295.
        His equation 12.a. We put sin(theta)=1, cos(theta)=0.
        """
        sq2   = np.sqrt(2.)
        pi    = np.pi
        gm    = self.gmring
        rring = self.rring
        xi    = rp/rring
        mu    = 2*xi/(1+xi**2)
        fxtot = -(gm/(pi*rring**2))*(xi*(pi-(pi-sq2)*mu) - sq2*mu)/((1.0-mu)*(1+xi**2)**1.5)
        return fxtot
        
    def make_grid_linlog_logrefine(self,rmax,r0,nr,drref,nrin=None):
        """
        Make a smooth radial grid between 0 and rmax such that
        it is quasi linear inward of r0, quasi logarithmic outside
        of r0 and log-refined toward r0 on both sides. The
        exact radius r0 is avoided. 

        Example:

        import numpy as np
        import matplotlib.pyplot as plt
        from ringgrav import *
        rmax  = 2000.
        r0    = 1.
        nr    = 1000
        drref = 0.01
        ring  = ringgrav()
        r     = ring.make_grid_linlog_logrefine(rmax,r0,nr,drref)
        dr    = r[1:] - r[:-1]
        plt.plot(r)
        plt.yscale('log')
        plt.show()
        """
        dum   = np.log(rmax/r0)
        if nrin is None:
            nrout = int(nr*dum/(dum+np.log(2.)))
            nrin  = nr-nrout
        else:
            nrout = nr-nrin
        lx0   = np.log(np.abs(drref/r0))
        lx1   = np.log(np.abs((rmax-r0)/r0))
        xx    = lx0 + (lx1-lx0)*np.linspace(0.,1.,nrout)
        exp   = np.exp(xx)
        dexp  = 0.5*(exp[1]-exp[0])
        dum   = 1+(exp[0]-dexp)/(1-exp[0])
        exp   = 1-(1-exp)*dum
        rout  = (1.0+exp)*r0
        xx    = lx0 + (0.-lx0)*np.linspace(0.,1.,nrin)
        exp   = np.exp(xx)
        dexp  = 0.5*(exp[1]-exp[0])
        dum   = 1+(exp[0]-dexp)/(1-exp[0])
        exp   = 1-(1-exp)*dum
        rin   = (1.0-exp)*r0
        r     = np.hstack((rin[::-1],rout))
        return r
    
    def make_ringgrav_table(self,rmax,nr=1000,drref=0.01,approx=None,nrin=None):
        """
        Because the computation of the ring gravity force can be somewhat
        time-consuming if it has to be done for a million pairs of radii,
        this method pre-computes these values for a set of radii.

        EXAMPLE AND TEST:
        
        import numpy as np
        import matplotlib.pyplot as plt
        from ringgrav import *
        rmax   = 100.
        nr     = 1000
        drref  = 0.01
        rring  = 1.
        gmring = 1.
        ring   = ringgrav(rring=rring,gmring=gmring)
        ring.make_ringgrav_table(rmax,nr=nr,drref=drref,approx='zypman')
        fzyp   = ring.table_f*1.0
        ring.make_ringgrav_table(rmax,nr=nr,drref=drref)
        ffull  = ring.table_f*1.0
        grho   = gmring/(2*np.pi*rring)
        fline  = 2*grho/(rring-ring.table_r)
        fpoint = -gmring/ring.table_r**2
        plt.plot(ring.table_r,np.abs(ffull),label='numeric',linewidth=3)
        plt.plot(ring.table_r,np.abs(fline),label='line')
        plt.plot(ring.table_r,np.abs(fpoint),label='point')
        plt.plot(ring.table_r,np.abs(fzyp),label='approx (Zypman)')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(ring.table_r[1],1e2)
        plt.ylim(bottom=1e-4)
        plt.xlabel('r')
        plt.ylabel('f')
        plt.legend()
        plt.show()

        The test shows that the correct limiting cases are reproduced.
        It also shows that the Zypman analytic approximation is fairly good,
        but underestimates the force for r<<R. This is, however, likely to
        be unimportant.
        """
        assert rmax>1.5*self.rring, "Error: rmax must be > 1.5*rring"
        rring  = self.rring
        gmring = self.gmring
        self.table_r = self.make_grid_linlog_logrefine(rmax,rring,nr,drref,nrin=nrin)
        self.table_f = np.zeros(nr)
        for ir in range(nr):
            if approx is None:
                self.table_f[ir] = self.ringgrav(self.table_r[ir])
            elif approx=='zypman':
                self.table_f[ir] = self.approx_ringgrav_zypman(self.table_r[ir])
            else:
                raise ValueError('Do not know approximation type')
        
    def make_ringgrav_interpol(self):
        """
        Make an interpolation function self.ringgrav_tabulated(). You must
        first call self.make_ringgrav_table().

        EXAMPLE:

        import numpy as np
        import matplotlib.pyplot as plt
        from ringgrav import *
        rmax   = 1e6
        nr     = 2000
        nrin   = 200
        drref  = 0.01
        ring   = ringgrav()
        ring.make_ringgrav_table(rmax,nr=nr,drref=drref,nrin=nrin)
        ring.make_ringgrav_interpol()
        ri     = 0.1*100**np.linspace(0.,1.,100)
        fi     = ring.ringgrav_interpol(ri)
        plt.plot(ring.table_r,np.abs(ring.table_f),label='table')
        plt.plot(ri,np.abs(fi),'o',label='interpol')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(ring.table_r[1],1e2)
        plt.ylim(bottom=1e-4)
        plt.xlabel('r')
        plt.ylabel('f')
        plt.legend()
        plt.show()

        SCALING:

        Note that this works best if you compute the table for dimensionless
        units (rring=1, gmring=1), then scale like this:

          f_real = (gmring_real/rring_real**2) * ring.ringgrav_interpol(ri/rring_real)

        Example:

        import numpy as np
        import matplotlib.pyplot as plt
        from ringgrav import *
        rmax   = 1e6
        nr     = 2000
        nrin   = 200
        drref  = 0.01
        ring   = ringgrav()
        ring.make_ringgrav_table(rmax,nr=nr,drref=drref,nrin=nrin)
        ring.make_ringgrav_interpol()
        ri     = 0.1*100**np.linspace(0.,1.,100)
        gmring_real = 50.
        rring_real  = 3.
        f_real = (gmring_real/rring_real**2) * ring.ringgrav_interpol(ri/rring_real)
        ring.gmring = gmring_real
        ring.rring  = rring_real
        f_check = np.zeros(len(ri))
        for ir in range(len(ri)):
            f_check[ir] = ring.ringgrav(ri[ir])
        plt.plot(ri,np.abs(f_real),label='real')
        plt.plot(ri,np.abs(f_check),'o',label='check')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(ring.table_r[1],1e2)
        plt.xlabel('r')
        plt.ylabel('f')
        plt.legend()
        plt.show()
        """
        assert hasattr(self,'table_f'), 'Error: Must first call self.make_ringgrav_table()'
        self.ringgrav_interpol = interp1d(self.table_r, self.table_f)

    def make_selfgrav_matrix(self,r,Ggrav=6.673e-08):
        """
        For self-gravity computation of an axisymmetric disk, this method computes
        the matrix such that fvector = Matrix * Sigmavector, where Sigmavector is a
        vector consisting of the surface densities and fvector is the force vector.

        EXAMPLE:

        import numpy as np
        import matplotlib.pyplot as plt
        from ringgrav import *
        au   = 1.49598e13     # AU [cm]
        MS   = 1.98892e+33    # Solar mass [g]
        GG   = 6.673e-08      # Grav const [GCS]
        nr   = 1000
        r    = 0.1*(1e5**np.linspace(0.,1.,nr))*au
        ring = ringgrav()
        ring.make_selfgrav_matrix(r,Ggrav=GG)
        # Now test with a delta function in sigma
        ir   = 700
        rring= r[ir]
        sig  = np.zeros(nr)
        mass = 1e-6*MS
        ri   = 0.5*(r[ir]+r[ir-1])
        ro   = 0.5*(r[ir]+r[ir+1])
        srf  = np.pi*(ro**2-ri**2)
        sig[ir] = mass/srf
        f    = np.dot(ring.matrix,sig)
        f[ir] = float('NaN')   # To make the plot nicer
        plt.plot(r,np.abs(f),linewidth=3,label='With matrix')
        plt.plot(ring.table_r*rring,np.abs(ring.table_f*GG*mass/rring**2),label='Direct')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(top=1e18)
        plt.ylim(bottom=1e-16)
        plt.legend()
        plt.show()
        """
        nr   = len(r)
        ri   = 0.5 * ( r[1:] + r[:-1] )
        surf = np.pi * ( ri[1:]**2 - ri[:-1]**2 )
        si   = np.pi * ( ri[0]**2 - r[0]**2 )
        so   = np.pi * ( r[-1]**2 - ri[-1]**2 )
        surf = np.hstack((si,surf,so))
        self.make_ringgrav_table(rmax=1.001*r.max()/r.min(),nr=2000,drref=0.01,nrin=200)
        self.make_ringgrav_interpol()
        self.matrix = np.zeros((nr,nr))
        for ir in range(nr):     # Loop over rows
            rk  = r[ir]/r        # Test particle fixed, ring radius = r vector
            ff  = self.ringgrav_interpol(rk)   # Dimensionless gravity
            ff *= surf           # Mult with surface, because need mass from Sigma
            ff *= Ggrav/r**2     # Mult by G/rring**2
            self.matrix[ir,:]  = ff
            self.matrix[ir,ir] = 0.0
