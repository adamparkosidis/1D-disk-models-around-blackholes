from snippet_header import DiskRadialModel, Disk2D, plt, MS, LS, au, finalize
mstar  = 1 * MS
lstar  = 1 * LS
dtg    = 0.01
kapdst = 1e2
sig0   = 2.0
r0     = 10*au
flang  = 0.05
zrmax  = 0.2
nr     = 30
opac   = ['supersimple', {'dusttogas': 0.01, 'kappadust': 1e2}]
disk   = DiskRadialModel(mstar=mstar, lstar=lstar, nr=nr)
disk.make_disk_from_simplified_lbp(sig0, r0, 1)
disk2d = Disk2D(disk, zrmax=0.3, meanopacitymodel=opac)

for vert in disk2d.verts:
    vert.iterate_vertical_structure()

disk2d.convert_1p1d_to_cyl2d(rhogas=True,tgas=True)

# plotting

plt.figure()
plt.plot(disk.r/au,disk2d.cyl2d_tgas[:,0],label='1+1D model')
plt.plot(disk.r/au,disk.tmid,label='1D model')
plt.xscale('log')
plt.xlabel('r [au]')
plt.yscale('log')
plt.ylabel(r'$T_{midplane} [K]$')
plt.text(2e1,6e1,'Where 1+1D fails...')
plt.legend()

finalize()
