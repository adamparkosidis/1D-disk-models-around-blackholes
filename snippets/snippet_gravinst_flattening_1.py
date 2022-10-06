from snippet_header import DiskRadialModel, np, plt, MS, LS, year, au, finalize
from copy import deepcopy

nr     = 1000
rout   = 1000 * au
rdisk  = 400 * au
mstar  = MS
lstar  = 10*LS
flang  = 0.02
rdisk  = 30*au
mdisk  = 0.25*MS
nudisk = rdisk**2/(1e6*year)
gam    = 1.0
time   = 0.

d      = DiskRadialModel(rout=rout, nr=nr, mstar=mstar,lstar=lstar,flang=flang)
d.make_disk_from_lyndenbellpringle(mdisk, rdisk, nudisk, gam, time)
d.add_dust(agrain=0.001)
d.compute_qtoomre()
dorig  = deepcopy(d)

d.gravinstab_flattening()
d.dust[0].gravinstab_apply_flattening()

plt.figure()
plt.loglog(dorig.r/au,dorig.sigma*dorig.r**2/MS,label='Gas (original)')
plt.loglog(d.r/au,d.sigma*d.r**2/MS,label='Gas (flattened)')
plt.loglog(dorig.r/au,dorig.dust[0].sigma*d.r**2/MS,'--',label='Dust (original)')
plt.loglog(d.r/au,d.dust[0].sigma*d.r**2/MS,'--',label='Dust (flattened)')
plt.xlim(left=0.1,right=300)
plt.ylim(bottom=1e-5,top=1e-1)
plt.xlabel('r [au]')
plt.ylabel(r'$\Sigma\,r^2\,[M_{\odot}]$')
plt.legend()

plt.figure()
plt.semilogx(dorig.r/au,dorig.qtoomre,label='Gas (original)')
plt.semilogx(d.r/au,d.qtoomre,label='Gas (flattened)')
plt.xlim(left=0.1,right=300)
plt.ylim(bottom=0e0,top=1e1)
plt.xlabel('r [au]')
plt.ylabel(r'$Q_{\mathrm{Toomre}}$')
plt.legend()

finalize(results=(d.sigma))
