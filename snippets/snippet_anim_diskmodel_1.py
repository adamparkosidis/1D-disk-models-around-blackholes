from snippet_header import np, DiskRadialModel, au, year, finalize

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

d.anim(time, sigma_array, ymin=1e-5, pause=30)

finalize([])
