# Model of TW Hydra, according to the paper
# Powell, Murray-Clay & Schlichtung 2017 (henceforth PMCS)
# They find that they can solve the radial drift problem if they
# increase the disk mass by a factor of a few hundred.
# Here this is tested with DISKLAB. We take a=9mm.
#
# Overall we can reproduce their results.
#
from disklab.diskradial import DiskRadialModel
from disklab.natconst import *
import matplotlib.pyplot as plt
mstar  = 0.8*MS     # Stellar mass (See PMCS Section 2)
lstar  = 0.28*LS    # Stellar luminosity (See PMCS Section 2)
tstart = 1e3*year
tend   = 5e6*year   # Stellar age assumption (See PMCS Section 2)
ntime  = 100        # Increase this for more accuracy
nr     = 100
time   = tstart * (tend/tstart)**(np.linspace(0.,1.,ntime+1))
d      = DiskRadialModel(rout=1000*au,nr=nr,mstar=mstar,lstar=lstar)
t0     = 82.        # PMCS: T_0=82K at r_0=1au (PMCS Eq.2)
d.tmid = t0*(d.r/au)**(-3./7.)  # PMCS Eq.1
d.compute_cs_and_hp()           # With new Tmid, recompute c_s and H_p
#sigc   = 0.5        # From Rosenfeld et al. (2012)
sigc   = 10.**2.5   # Estimate from PMCS (Their page 5, right col, below Eq.8)
rc     = 30*au      # From Rosenfeld et al. (2012)
gam    = 1.         # From Rosenfeld et al. (2012)
d.make_disk_from_simplified_lbp(sigc,rc,gam)
agrain = 0.9        # PMCS largest grain assumption = 9 mm (see their Section 3)
xigrain= 2.0        # PMCS material density (between their Eq.2 and Eq.3)
d.add_dust(agrain=agrain,xigrain=xigrain)
for itime in range(1,ntime+1):
   d.dust[0].compute_dust_radial_drift_next_timestep(time[itime]-time[itime-1],fixgas=True)
plt.figure(1)
d.plot(d.sigma,label='gas',ylabel=r'$\Sigma$')
d.plot(d.dust[0].sigma,oplot=True,label='dust')
plt.legend()
plt.figure(2)
d.compute_qtoomre()
d.plot(d.qtoomre,ylabel=r'$Q_{\mathrm{Toomre}}$',ymax=1e2)
