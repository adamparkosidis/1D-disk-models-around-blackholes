from snippet_header import DiskRadialModel, Disk2D, plt, MS, LS, au, finalize
import copy
mstar  = 1 * MS
lstar  = 1 * LS
sig0   = 10.
r0     = 50. * au
dtg    = 0.01
kapdst = 1e2
flang  = 0.025
zrmax  = 1.0
nr     = 100
rin    = 1 * au
rout   = 500 * au
alpha  = 1e-6
opac   = ['supersimple', {'dusttogas': 0.01, 'kappadust': 1e2}]
disk   = DiskRadialModel(mstar=mstar, lstar=lstar, nr=nr, rin=rin, rout=rout, alpha=alpha)
disk.make_disk_from_simplified_lbp(sig0, r0, 0.5)
disk2d = Disk2D(disk, zrmax=zrmax, meanopacitymodel=opac)

for vert in disk2d.verts:
    vert.iterate_vertical_structure()

disk11d = copy.deepcopy(disk2d)

maxiter = 1
for iter in range(maxiter):
    print("Vertical structure iteration {}".format(iter))
    for vert in disk2d.verts:
        vert.compute_mean_opacity()
    disk2d.radial_raytrace()
    disk2d.setup_spherical_coordinate_system()
    disk2d.solve_2d_rad_diffusion(linsol_convcrit=1e-6, linsol_itermax=2000,
                                  nonlin_convcrit=1e-3, nonlin_itermax=20,
                                  simplecoordtrans=False)
    for vert in disk2d.verts:
        vert.compute_rhogas_hydrostatic()

plt.figure()
for ir in range(0, nr, 10):
    vert = disk2d.verts[ir]
    plt.plot(vert.z / vert.r, vert.rhogas)
    vert = disk11d.verts[ir]
    plt.plot(vert.z / vert.r, vert.rhogas, '--')
plt.xlabel('z/r')
plt.yscale('log')
plt.ylim(bottom=1e-26)
plt.ylabel(r'$\rho [\mathrm{g}/\mathrm{cm}^3]$')
plt.savefig('fig_snippet_vertstruct_2d_3_1.pdf')

plt.figure()
for ir in range(0, nr, 10):
    vert = disk2d.verts[ir]
    plt.plot(vert.z / vert.r, vert.tgas)
    vert = disk11d.verts[ir]
    plt.plot(vert.z / vert.r, vert.tgas, '--')
plt.xlabel('z/r')
plt.ylabel(r'$T [K]$')
plt.ylim(top=1e3, bottom=3e0)
plt.yscale('log')
plt.savefig('fig_snippet_vertstruct_2d_3_2.pdf')

finalize()
