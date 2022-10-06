from snippet_header import DiskRadialModel, finalize, year, au, Mea, MS, np

tstart   = 1e4 * year
tend     = 2e6 * year
ntime    = 100
nr       = 2000
alpha    = 1e-2
agrain   = 1e-2
mplanet1 = 100 * Mea
aplanet1 = 5 * au
mplanet2 = 200 * Mea
aplanet2 = 20 * au
mdisk0   = 1e-2 * MS
rdisk0   = 10 * au

d = DiskRadialModel(rin=0.1 * au, rout=1000 * au, nr=nr)
d.make_disk_from_lbp_alpha(mdisk0, rdisk0, alpha, tstart)
d.add_dust(agrain=agrain)
d.add_planet_gap(aplanet1, 'duffell', mpl=mplanet1, smooth=2., log=True, innu=True)
d.add_planet_gap(aplanet2, 'duffell', mpl=mplanet2, smooth=2., log=True, innu=True)

time = tstart * (tend / tstart)**(np.linspace(0., 1., ntime + 1))
sigmadust_array = np.zeros((ntime + 1, len(d.r)))
sigmadust_array[0, :] = d.dust[0].sigma.copy()
for itime in range(1, ntime + 1):
    dt = time[itime] - time[itime - 1]
    d.compute_viscous_evolution_and_dust_drift_next_timestep(dt)
    sigmadust_array[itime, :] = d.dust[0].sigma.copy()

d.anim(time, sigmadust_array, ymin=1e-5, pause=30,
       ylabel=r'$\Sigma_{\mathrm{dust}}$')

finalize([])
