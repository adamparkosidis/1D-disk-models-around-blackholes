from snippet_header import DiskRadialModel, Disk2D, np, plt, MS, LS, au, finalize

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
opac = ['supersimple', {'dusttogas': 0.01, 'kappadust': 1e2}]
disk = DiskRadialModel(mstar=mstar, lstar=lstar, nr=nr, rin=rin, rout=rout, alpha=alpha)
disk.make_disk_from_simplified_lbp(sig0, r0, 0.5)
disk2d = Disk2D(disk, zrmax=zrmax, meanopacitymodel=opac)

for vert in disk2d.verts:
    vert.iterate_vertical_structure()

disk11d = copy.deepcopy(disk2d)

maxiter = 10
for iter in range(maxiter):
    print("Vertical structure iteration {}".format(iter))
    for vert in disk2d.verts:
        vert.compute_mean_opacity()
    disk2d.radial_raytrace()
    disk2d.solve_2d_rad_diffusion(linsol_convcrit=1e-6, linsol_itermax=2000,
                                  nonlin_convcrit=1e-3, nonlin_itermax=20,
                                  simplecoordtrans=True)
    for vert in disk2d.verts:
        vert.compute_rhogas_hydrostatic()

xcoord = np.log10(disk2d.r / au)
ycoord = np.pi / 2 - disk2d.spher_theta
extent = [xcoord.min(), xcoord.max(), ycoord.min(), ycoord.max()]
skip = 4
sl1d = slice(None, None, skip)
sl2d = (slice(None, None, skip), slice(None, None, -skip))

plt.figure()
plt.imshow(np.log10(disk2d.spher_rt_s[:, :, 0].T + 1e-30),
           extent=extent, aspect='auto')
plt.xlabel(r'$^{10}\log(r/\mathrm{au})$')
plt.ylabel(r'$z/r$')
plt.title('Source term')
plt.colorbar()

plt.figure()
plt.imshow(np.log10(disk2d.spher_rt_t[:, :, 0].T), extent=extent, aspect='auto')
plt.xlabel(r'$^{10}\log(r/\mathrm{au})$')
plt.ylabel(r'$z/r$')
plt.title('Temperature')
plt.colorbar()

plt.figure()
plt.plot(disk2d.disk.r / au, disk2d.spher_rt_t[:, 89, 0])
plt.plot(disk2d.disk.r / au, disk2d.spher_rt_t[:, 0, 0])
plt.ylabel('T [K]')
plt.xlabel('r [au]')
plt.xscale('log')
plt.yscale('log')

plt.figure()
plt.plot(disk2d.disk.r / au, disk2d.spher_rt_jdiff[:, 89, 0])
plt.plot(disk2d.disk.r / au, disk2d.spher_rt_jdiff[:, 0, 0])
plt.ylabel('J_{diff}')
plt.xlabel('r [au]')
plt.xscale('log')
plt.yscale('log')

plt.figure()
plt.imshow(np.log10(disk2d.spher_rt_jdiff[:, :, 0].T), extent=extent, aspect='auto')
plt.xlabel(r'$^{10}\log(r/\mathrm{au})$')
plt.ylabel(r'$z/r$')
dirvec_r = disk2d.spher_rt_hflux[0][:, ::-1, 0] / disk2d.spher_rt_jdiff[:, ::-1, 0]
dirvec_t = -disk2d.spher_rt_hflux[1][:, ::-1, 0] / disk2d.spher_rt_jdiff[:, ::-1, 0]
plt.quiver(xcoord[sl1d], ycoord[sl1d], dirvec_r[sl2d].T,
           dirvec_t[sl2d].T, units='xy')
plt.title('Mean intensity and vec H/J')

plt.figure()
plt.imshow(
    np.log10(disk2d.spher_rt_fld_limiter[:, :, 0].T), extent=extent, aspect='auto')
plt.xlabel(r'$^{10}\log(r/\mathrm{au})$')
plt.ylabel(r'$z/r$')
plt.title('Flux limiter')
plt.colorbar()

finalize()
