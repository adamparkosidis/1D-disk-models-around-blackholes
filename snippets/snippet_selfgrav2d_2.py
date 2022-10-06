from snippet_header import np, DiskRadialModel, Disk2D, plt, MS, LS, au, GG, surface, finalize
from copy import deepcopy

mstar = MS
mdisk = 0.6e-1*MS
plsig = -1.
rcut  = 20*au
plcut = -2.
rcti  = 2.*au
plcti = 4.
sigflr= 1e-5

d         = DiskRadialModel()
r         = d.r
d.sigma   = 1.0 * (r/rcut)**plsig * np.exp(-(rcut/r)**plcut) * np.exp(-(rcti/r)**plcti)
d.compute_mass()
d.sigma  *= mdisk/d.mass
d.sigma  += sigflr
d.compute_mass()
d.tmid    = 150*(r/au)**(-0.5)
d.compute_cs_and_hp()
d.compute_qtoomre()

disk      = Disk2D(d,irradmode='isothermal')
disk.compute_vphi()
disksg    = deepcopy(disk)
disksg.setup_spherical_coordinate_system(nr_thetapadding=10,theta0=0.01,nr_rpadding_in=9,
                                          rin_factor=0.5,rout_factor=100.,nr_rpadding=30)
diskiterations = []
for iter in range(4):
    print iter
    disksg.solve_2d_selfgrav()
    for v in disksg.verts:
        v.compute_rhogas_hydrostatic(add_selfgrav=True)
    disksg.compute_vphi(add_selfgrav=True)
    diskiterations.append(deepcopy(disksg))

ir=65
plt.figure()
plt.plot(disk.cyl2d_zr[ir,:],disk.verts[ir].rhogas[:],label='No SG')
plt.plot(disk.cyl2d_zr[ir,:],diskiterations[-1].cyl2d_rhogas[ir,:],label='2-D SG')
plt.xlabel('z/r')
plt.ylabel(r'$\rho\;[g/cm^3]$')
plt.title('Vertical density structure at r= {0:6.2f} au'.format(d.r[ir]/au))
plt.legend()

plt.figure()
plt.plot(disk.cyl2d_zr[ir,:],disk.verts[ir].vphi[:]/1e5,label='No SG')
plt.plot(disk.cyl2d_zr[ir,:],diskiterations[-1].verts[ir].vphi[:]/1e5,label='2-D SG')
plt.xlabel('z/r')
plt.ylabel(r'$v_\phi\;[km/s]$')
plt.title('Azimuthal velocity at r= {0:6.2f} au'.format(d.r[ir]/au))
plt.legend()

cumulmass = np.zeros(len(d.r)+1)
cumulmass[0] = mstar
ri = np.hstack((d.r[0],d.get_ri(),d.r[-1]))
for ir in range(1,len(d.r)+1):
    cumulmass[ir] = cumulmass[ir-1] + d.sigma[ir-1]*np.pi*(ri[ir]**2-ri[ir-1]**2)
vphi_approx = np.sqrt(GG*cumulmass[:-1]/d.r)
plt.figure()
plt.semilogx(disk.r/au,(diskiterations[-1].vphi[:,0]+1e-25)/(disk.vphi[:,0]+1e-30),label='2-D SG')
plt.semilogx(disk.r/au,(vphi_approx+1e-25)/(disk.vphi[:,0]+1e-30),label='SG Approx')
plt.ylim(0.9,1.2)
plt.xlabel(r'$r\;[au]$')
plt.ylabel(r'$v_\phi(SG)/v_\phi(noSG)$')
plt.legend()

finalize()
