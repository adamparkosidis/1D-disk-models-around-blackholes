import numpy as np
import copy

def explicitadvdiff3d(geom, x, y, z, vx, vy, vz, diffcoef, q, dt,
                      wcoef=None, geomfactors=None, checkcfl=True,
                      stepcfl=False,nstepmax=100):
    #
    # Compute the geometric factors
    #
    nn = q.shape
    if geomfactors is not None:
        xx,yy,zz,dx,dy,dz,dxi,dyi,dzi,gmc,gmi,gmdi = geomfactors
    else:
        xx,yy,zz,dx,dy,dz,dxi,dyi,dzi,gmc,gmi,gmdi = get_geomfactors(geom, x, y, z)
    #
    # Put things in convenient lists
    #
    #xxc  = [xx,yy,zz]
    dxxc = [dx,dy,dz]
    dxxi = [dxi,dyi,dzi]
    vv   = [vx,vy,vz]
    if len(diffcoef)==3:
        dc = diffcoef
    else:
        dc = [diffcoef,diffcoef,diffcoef]
    dci = [0.,0.,0.]
    if nn[0]>1: dci[0] = 0.5 * ( dc[0][1:,:,:] + dc[0][:-1,:,:] )
    if nn[1]>1: dci[1] = 0.5 * ( dc[1][:,1:,:] + dc[1][:,:-1,:] )
    if nn[2]>1: dci[2] = 0.5 * ( dc[2][:,:,1:] + dc[2][:,:,:-1] )
    #
    # Check CFL
    #
    dtuse = dt
    nstep = 1
    cflsafe = 0.5
    if checkcfl or stepcfl:
        dtcfl = cflsafe*get_cfl_timestep(vx,vy,vz,dc,geomfactors=geomfactors)
        if dt > dtcfl and not stepcfl: return False
        if stepcfl:
            if dtcfl>=dt:
                dtuse = dt
                nstep = 1
            else:
                dtuse = dtcfl
                nstep = int(dt/dtuse) + 1
    #
    # Advection in each direction
    #
    time = 0.
    for istep in range(nstep):
        if time+dtuse>dt and stepcfl: dtuse = max(0.,dt-time)
        for idir in range(3):
            if nn[idir]>1:
                vi      = get_vi(vv[idir],idir)
                fi_adv  = get_fluxi_from_donorcell(q,vi,idir)
                fi_diff = get_fluxi_from_diffusion(q,dci[idir],idir,gmdi[idir],dxxi[idir],w=wcoef)
                fi      = fi_adv + fi_diff
                update_quantity_with_fluxi_with_geometry(q,fi,idir,gmc[idir],gmi[idir],dxxc[idir],dtuse)
        time += dtuse
    return True

def get_geomfactors(geom, x, y, z):
    nx = len(x)
    ny = len(y)
    nz = len(z)
    xx,yy,zz = np.meshgrid(x,y,z,indexing='ij')
    xxi = None
    yyi = None
    zzi = None
    if nx>1: xxi = 0.5 * ( xx[1:,:,:] + xx[:-1,:,:] )
    if ny>1: yyi = 0.5 * ( yy[:,1:,:] + yy[:,:-1,:] )
    if nz>1: zzi = 0.5 * ( zz[:,:,1:] + zz[:,:,:-1] )
    dxi = None
    dyi = None
    dzi = None
    if nx>1: dxi = xx[1:,:,:] - xx[:-1,:,:]
    if ny>1: dyi = yy[:,1:,:] - yy[:,:-1,:]
    if nz>1: dzi = zz[:,:,1:] - zz[:,:,:-1]
    dx = np.zeros_like(xx)
    dy = np.zeros_like(yy)
    dz = np.zeros_like(zz)
    if nx>1:
        dx[1:-1,:,:] = xxi[1:,:,:] - xxi[:-1,:,:]
        dx[0,:,:]    = dx[1,:,:]
        dx[-1,:,:]   = dx[-2,:,:]
    if ny>1:
        dy[:,1:-1,:] = yyi[:,1:,:] - yyi[:,:-1,:]
        dy[:,0,:]    = dy[:,1,:]
        dy[:,-1,:]   = dy[:,-2,:]
    if nz>1:
        dz[:,:,1:-1] = zzi[:,:,1:] - zzi[:,:,:-1]
        dz[:,:,0]    = dz[:,:,1]
        dz[:,:,-1]   = dz[:,:,-2]
    if geom=='cartesian':
        gmc  = []
        gmc.append(np.ones_like(xx))
        gmc.append(np.ones_like(yy))
        gmc.append(np.ones_like(zz))
        gmi = []
        gmi.append(np.ones_like(xxi))
        gmi.append(np.ones_like(yyi))
        gmi.append(np.ones_like(zzi))
        gmdi = []
        gmdi.append(np.ones_like(xxi))
        gmdi.append(np.ones_like(yyi))
        gmdi.append(np.ones_like(zzi))
    elif geom=='spherical':      # r,theta,phi
        gmc  = []
        gmc.append(xx**2)
        gmc.append(xx*np.sin(yy))
        gmc.append(xx*np.sin(yy))
        gmi = []
        gmi.append(xxi**2)
        gmi.append(np.sin(yyi))
        gmi.append(np.ones_like(zzi))
        gmdi = []
        gmdi.append(np.ones_like(xxi))
        gmdi.append(xx[:,:-1,:])
        gmdi.append(xx[:,:,:-1]*np.sin(yy[:,:,:-1]))
    elif geom=='cylindrical':    # r,z,phi
        gmc  = []
        gmc.append(xx)
        gmc.append(np.ones_like(yy))
        gmc.append(xx)
        gmi = []
        gmi.append(xxi)
        gmi.append(np.ones_like(yyi))
        gmi.append(np.ones_like(zzi))
        gmdi = []
        gmdi.append(np.ones_like(xxi))
        gmdi.append(np.ones_like(yyi))
        gmdi.append(xx[:,:,:-1])
    elif geom=='cylindricalzconical':    # r,z,phi with z_i=zeta_i*r
        gmc  = []
        gmc.append(xx**2)
        gmc.append(np.ones_like(yy))
        gmc.append(xx)
        gmi = []
        gmi.append(xxi**2)
        gmi.append(np.ones_like(yyi))
        gmi.append(np.ones_like(zzi))
        gmdi = []
        gmdi.append(np.ones_like(xxi))
        gmdi.append(np.ones_like(yyi))
        gmdi.append(xx[:,:,:-1])
    else:
        raise ValueError('Geometry not known')
    return xx,yy,zz,dx,dy,dz,dxi,dyi,dzi,gmc,gmi,gmdi

def get_vi(v,idir):
    """
    Returns the interface velocity of a fluid in direction idir.
    """
    if idir==0:
        vi = 0.5 * ( v[1:,:,:] + v[:-1,:,:] )
    elif idir==1:
        vi = 0.5 * ( v[:,1:,:] + v[:,:-1,:] )
    elif idir==2:
        vi = 0.5 * ( v[:,:,1:] + v[:,:,:-1] )
    else: raise ValueError("Wrong idir")
    return vi

def get_fluxi_from_donorcell(q,vi,idir):
    """
    Returns the interface fluxes in direction idir,
    given the interface velocities vi, using donor-cell.
    The quantity to be advected is linked by q.
    """
    assert len(q.shape)==3, "Arrays in hydro must always be 3D (Tip: Python tends to collapse dimensions with 1; avoid that at all cost)"
    assert len(vi.shape)==3, "Arrays in hydro must always be 3D (Tip: Python tends to collapse dimensions with 1; avoid that at all cost)"
    nn    = q.shape
    nm    = [nn[0],nn[1],nn[2]]
    nm[idir] -= 1
    fi    = np.zeros(nm)
    ipos  = np.where(vi>0.)
    ineg  = np.where(vi<0.)
    if len(ipos[0])>0:
        fi[ipos]       += vi[ipos]*q[ipos]
    if len(ineg[0])>0:
        inegp           = copy.deepcopy(ineg)
        inegp[idir][:] += 1
        fi[ineg]       += vi[ineg]*q[inegp]
    return fi

def get_fluxi_from_diffusion(q,di,idir,gmdi,dxi,w=None):
    """
    Returns the interface diffusion fluxes in direction idir,
    given the interface diffusion coefficient di, 
    """
    assert len(q.shape)==3, "Arrays in hydro must always be 3D (Tip: Python tends to collapse dimensions with 1; avoid that at all cost)"
    assert len(di.shape)==3, "Arrays in hydro must always be 3D (Tip: Python tends to collapse dimensions with 1; avoid that at all cost)"
    if w is None:
        w = np.ones_like(q)
    qw    = q*w
    if idir==0:
        wi    = 0.5 * ( w[1:,:,:] + w[:-1,:,:] )
        wdqw  = wi * ( qw[1:,:,:] - qw[:-1,:,:] ) / ( dxi * gmdi )
    elif idir==1:
        wi    = 0.5 * ( w[:,1:,:] + w[:,:-1,:] )
        wdqw  = wi * ( qw[:,1:,:] - qw[:,:-1,:] ) / ( dxi * gmdi )
    elif idir==2:
        wi    = 0.5 * ( w[:,:,1:] + w[:,:,:-1] )
        wdqw  = wi * ( qw[:,:,1:] - qw[:,:,:-1] ) / ( dxi * gmdi )
    else: raise ValueError("Wrong idir")
    fi    = -di*wdqw
    return fi

def update_quantity_with_fluxi_with_geometry(q,fi,idir,gmc,gmi,dx,dt):
    """
    Spherical or cylindrical version of update_quantity_with_fluxi.
    For these coordinates the divergence operator (which this routine
    is in fact) requires geometric factors, which are included here.
    """
    assert len(q.shape)==3, "Arrays in hydro must always be 3D (Tip: Python tends to collapse dimensions with 1; avoid that at all cost)"
    assert len(fi.shape)==3, "Arrays in hydro must always be 3D (Tip: Python tends to collapse dimensions with 1; avoid that at all cost)"
    if idir==0:
        q[1:-1,:,:] -= dt * ( fi[1:,:,:]*gmi[1:,:,:] - fi[:-1,:,:]*gmi[:-1,:,:] ) / ( dx[1:-1,:,:]*gmc[1:-1,:,:] )
    elif idir==1:
        q[:,1:-1,:] -= dt * ( fi[:,1:,:]*gmi[:,1:,:] - fi[:,:-1,:]*gmi[:,:-1,:] ) / ( dx[:,1:-1,:]*gmc[:,1:-1,:] )
    elif idir==2:
        q[:,:,1:-1] -= dt * ( fi[:,:,1:]*gmi[:,:,1:] - fi[:,:,:-1]*gmi[:,:,:-1] ) / ( dx[:,:,1:-1]*gmc[:,:,1:-1] )
    else: raise ValueError("Wrong idir")

def get_cfl_timestep(vx,vy,vz,diffcoef,geomfactors=None):
    nn = vx.shape
    if len(diffcoef)==3:
        dc = diffcoef
    else:
        dc = [diffcoef,diffcoef,diffcoef]
    if geomfactors is not None:
        xx,yy,zz,dx,dy,dz,dxi,dyi,dzi,gmc,gmi,gmdi = geomfactors
    else:
        xx,yy,zz,dx,dy,dz,dxi,dyi,dzi,gmc,gmi,gmdi = get_geomfactors(geom, x, y, z)
    gmdc = [np.ones_like(gmc[0]),np.ones_like(gmc[1]),np.ones_like(gmc[2])]
    if nn[0]>1: gmdc[0][1:-1,:,:] = 0.5 * ( gmdi[0][1:,:,:] + gmdi[0][:-1,:,:] )
    if nn[1]>1: gmdc[1][:,1:-1,:] = 0.5 * ( gmdi[1][:,1:,:] + gmdi[1][:,:-1,:] )
    if nn[2]>1: gmdc[2][:,:,1:-1] = 0.5 * ( gmdi[2][:,:,1:] + gmdi[2][:,:,:-1] )
    dtloc = [np.ones_like(gmc[0])*1e99,np.ones_like(gmc[1])*1e99,np.ones_like(gmc[2])*1e99]
    if nn[0]>1: dtloc[0][1:-1,:,:] = (dx*gmdc[0]/((np.abs(vx)+dc[0]/(dx*gmdc[0])+1e-90)))[1:-1,:,:]
    if nn[1]>1: dtloc[1][:,1:-1,:] = (dy*gmdc[1]/((np.abs(vy)+dc[1]/(dy*gmdc[1])+1e-90)))[:,1:-1,:]
    if nn[2]>1: dtloc[2][:,:,1:-1] = (dz*gmdc[2]/((np.abs(vz)+dc[2]/(dz*gmdc[2])+1e-90)))[:,:,1:-1]
    dt = min(dtloc[0].min(),dtloc[1].min(),dtloc[2].min())
    return dt
