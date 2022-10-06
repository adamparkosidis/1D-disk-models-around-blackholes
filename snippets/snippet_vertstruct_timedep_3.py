from snippet_header import DiskVerticalModel, plt, MS, year, au, finalize

mstar  = 1 * MS
r      = 1 * au
siggas = 1700.
dtg    = 0.01
kapdst = 1e2
flang  = 0.05
flang1 = 0.07
eps    = 1e-3
dt     = 1e2 * year   # Time step
opac = ['supersimple', {'dusttogas': 0.01, 'kappadust': 1e2}]
vert = DiskVerticalModel(mstar, r, siggas, flang=flang, zrmax=0.2, meanopacitymodel=opac)
vert.iterate_vertical_structure()

plt.figure()
plt.plot(vert.z / vert.r, vert.tgas, label='before')
vert.flang = flang1
vert.irradiate_with_flaring_index()
vert.timestep_vert_rad_diffusion((1 - eps) * dt)
plt.plot(vert.z / vert.r, vert.tgas, label='after timdep RT')
vert.compute_rhogas_hydrostatic_adiabatic()
plt.plot(vert.z / vert.r, vert.tgas, label='after adiab restat')
vert.compute_mean_opacity()
vert.irradiate_with_flaring_index()
vert.timestep_vert_rad_diffusion(eps * dt)
plt.plot(vert.z / vert.r, vert.tgas, label='after final RT')
plt.ylim(bottom=0)
plt.legend(loc='lower right')

finalize(results=(vert.rhogas,vert.tgas))
