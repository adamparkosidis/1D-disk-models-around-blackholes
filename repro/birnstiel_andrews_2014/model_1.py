# Model of Birnstiel & Andrews (2014), their Fig.1 and Fig 2
from disklab.diskradial import DiskRadialModel
from disklab.natconst import *
import matplotlib.pyplot as plt
mstar  = 0.8*MS     # Stellar mass (See B&A below Eq.8)
tstart = 1e3*year
tend   = 1e5*year   # Red curve of B&A Fig.2
ntime  = 100        # Increase this for more accuracy (e.g. to 1000)
nr     = 1000
time   = tstart * (tend/tstart)**(np.linspace(0.,1.,ntime+1))
d      = diskDiskRadialModelmodel(rin=5*au,rout=500*au,nr=nr,mstar=mstar)
d.alpha= 1e-3       # B&A (below Eq. 8)
t0     = 200.       # B&A: T_0=200K at r_0=1au
d.tmid = t0*(d.r/au)**(-0.5)  # B&A Eq.8
d.compute_cs_and_hp()         # With new Tmid, recompute c_s and H_p
sigc   = 1.         # First normalize to Sigma_c=1, then renormalize to Mdisk=0.01*MS
rc     = 20*au      # B&A choose r_c=20au (below Eq.8)
gam    = 1.         # B&A choose gamma=1
d.make_disk_from_simplified_lbp(sigc,rc,gam)
d.compute_mass()
mdisk  = 0.01*MS    # Disk mass B&A (below Eq.8)
d.sigma= d.sigma*mdisk/d.mass
d.compute_mass()    # Recompute the mass (just for self-consistency)
d.compute_rhomid_from_sigma() # When modifying Sigma, we must recompute rhomid
d.compute_vr_at_interfaces()  # We do not evolve Sigma_gas, but need v_r of gas
d.Sc   = 1e10       # Switch off dust diffusion while still viscously accreting (Note: somewhat dubious)
agrain = 1e-4       # B&A choose 1 micron grain radius (page 3, left col)
xigrain= 1.6        # B&A choose 1.6 g/cm^3 material density (below Eq.6)
d.add_dust(agrain=agrain,xigrain=xigrain)
sigd0=d.dust[0].sigma.copy()
for itime in range(1,ntime+1):
   d.dust[0].compute_dust_radial_drift_next_timestep(time[itime]-time[itime-1],fixgas=False)
plt.figure(1)
d.plot(d.dust[0].vr,ymin=-35,ymax=+15,ylog=False)
plt.autoscale(enable=True,axis='x',tight=True)
plt.figure(2)
d.plot(d.sigma,label='gas',ylabel=r'$\Sigma$',ymin=1e-6,ymax=1e1)
d.plot(d.dust[0].sigma,oplot=True,label='dust')
d.plot(sigd0,oplot=True,label='dust t=0')
plt.autoscale(enable=True,axis='x',tight=True)
plt.legend()
