from snippet_header import DiskRadialModel, Disk2D, plt, MS, LS, finalize
import copy
mstar  = 1 * MS
lstar  = 1 * LS
mdisk  = 0.01 * MS
dtg    = 0.01
kapdst = 1e2
flang  = 0.025
zrmax  = 0.2
nr     = 10
opac   = ['supersimple', {'dusttogas': 0.01, 'kappadust': 1e2}]
disk   = DiskRadialModel(mstar=mstar, lstar=lstar, mdisk=mdisk, nr=nr)
disk2d = Disk2D(disk, zrmax=0.3, meanopacitymodel=opac)

for vert in disk2d.verts:
    vert.iterate_vertical_structure()

disk2d_orig = copy.deepcopy(disk2d)

maxiter = 10
for iter in range(maxiter):
    for vert in disk2d.verts:
        vert.compute_mean_opacity()
    disk2d.radial_raytrace()
    for vert in disk2d.verts:
        vert.solve_vert_rad_diffusion()
        vert.compute_rhogas_hydrostatic()

# plotting

plt.figure()
for vert in disk2d.verts:
    plt.plot(vert.z / vert.r, vert.rhogas)
plt.xlabel('z/r')
plt.yscale('log')
plt.ylim(bottom=1e-26)
plt.ylabel(r'$\rho [\mathrm{g}/\mathrm{cm}^3]$')

plt.figure()
for vert in disk2d.verts:
    plt.plot(vert.z / vert.r, vert.tgas)
plt.xlabel('z/r')
plt.ylabel(r'$T [K]$')
plt.ylim(top=2e3, bottom=3e0)
plt.yscale('log')

finalize()
