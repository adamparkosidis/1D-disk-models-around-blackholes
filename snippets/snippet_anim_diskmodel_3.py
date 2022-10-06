from snippet_header import np, DiskRadialModel, au, year, finalize
from disklab.viewarr import *

d = DiskRadialModel(rout=1000 * au)
d.make_disk_from_simplified_lbp(1e2, 1 * au, 1.0)
tmax = 1e6 * year
nt = 100
time = np.linspace(0, tmax, nt)
sigma_array = np.zeros((nt, len(d.r)))
sigma_array[0, :] = d.sigma.copy()
for itime in range(1, nt):
    dt = time[itime] - time[itime - 1]
    d.compute_viscous_evolution_next_timestep(dt)
    sigma_array[itime, :] = d.sigma.copy()

viewarr(np.log10(sigma_array+1e-6),index=1,x=np.log10(d.r/au),idxnames=['t [Myr] = ','r [au]'],idxvals=[time/year/1e6,d.r/au],idxformat='9.2e')

finalize([])
