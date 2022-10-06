from snippet_header import np, DiskRadialModel, Disk2D, plt, MS, LS, au, GG, surface, finalize

mstar = MS
mdisk = 0.6e-1*MS
plsig = -1.
rcut  = 20*au
plcut = -2.
rcti  = 2.*au
plcti = 4.

d         = DiskRadialModel()
r         = d.r
d.sigma   = 1.0 * (r/rcut)**plsig * np.exp(-(rcut/r)**plcut) * np.exp(-(rcti/r)**plcti)
d.compute_mass()
d.sigma  *= mdisk/d.mass
d.compute_mass()
d.tmid    = 150*(r/au)**(-0.5)
d.compute_cs_and_hp()

disk      = Disk2D(d,irradmode='isothermal')
disk.setup_spherical_coordinate_system(nr_thetapadding=10,theta0=0.01,nr_rpadding_in=9,
                                          rin_factor=0.5,rout_factor=100.,nr_rpadding=30)
disk.solve_2d_selfgrav()

# For comparison/testing: Compute the plane-parallel version of the f_z

f_z_approx = np.zeros_like(disk.cyl2d_selfgrav_fz)
zz         = disk.cyl2d_zz
rho        = disk.cyl2d_rho
for iz in range(1,len(disk.cyl2d_selfgrav_fz[0,:])):
    f_z_approx[:,iz] = f_z_approx[:,iz-1] + 2*(zz[:,iz]-zz[:,iz-1])*rho[:,iz-1]
f_z_approx *= -2*np.pi*GG

# Plotting

stride = 4
lgr    = np.log10(disk.r/au)
zr     = disk.cyl2d_zz[0,:]/disk.r[0]
pot1d  = -GG*d.mass/disk.spher_r

plt.figure()
plt.loglog(disk.spher_r/au,-disk.spher_pot[:,1,0],'.',label='Spher: Pole')
plt.loglog(disk.spher_r/au,-disk.spher_pot[:,-2,0],'.',label='Spher: Equator')
plt.loglog(disk.spher_r/au,-pot1d,':',label='Spher: 1D')
plt.loglog(disk.cyl2d_r/au,-disk.cyl2d_pot[:,0],label='Cyl: Equator')
plt.xlabel('r [au]')
plt.ylabel(r'$-\Phi\;[erg/g]$')
plt.legend()

plt.figure()
plt.plot(np.pi/2-disk.spher_tt[9+45,:],disk.spher_pot[9+45,:],label='Spher')
plt.plot(disk.cyl2d_zz[45,:]/disk.cyl2d_rr[45,:],disk.cyl2d_pot[45,:],label='Cyl')
plt.xlabel(r'z/r (cyl) or $\pi/2-\theta$ (spher)')
plt.ylabel(r'$\Phi\;[erg/g]$')
plt.title('Potential at 3.1 au')
plt.legend()

plt.figure()
plt.plot(disk.cyl2d_zz[45,:]/disk.cyl2d_rr[45,:],disk.cyl2d_selfgrav_fz[45,:],label='Cyl 2D')
plt.plot(disk.cyl2d_zz[45,:]/disk.cyl2d_rr[45,:],f_z_approx[45,:],label='Cyl 1D')
plt.xlabel(r'z/r (cyl) or $\pi/2-\theta$ (spher)')
plt.ylabel(r'$f_z\;[dyne/g]$')
plt.title('Vertical body force at 3.1 au')
plt.legend()

fig,ax=surface(disk.cyl2d_pot,x=lgr,y=zr,cstride=stride,rstride=stride,
               xlabel=r'$^{10}$lg(r/au)',ylabel=r'$z/r$',zlabel=r'$\Phi\;[erg/g]$')

fig,ax=surface(disk.cyl2d_selfgrav_fr,x=lgr,y=zr,cstride=stride,rstride=stride,
               xlabel=r'$^{10}$lg(r/au)',ylabel=r'$z/r$',zlabel=r'$f_r\;[dyne/g]$')

fig,ax=surface(disk.cyl2d_selfgrav_fz,x=lgr,y=zr,cstride=stride,rstride=stride,
               xlabel=r'$^{10}$lg(r/au)',ylabel=r'$z/r$',zlabel=r'$f_z\;[dyne/g]$')

finalize()

