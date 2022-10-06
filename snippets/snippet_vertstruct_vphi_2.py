from snippet_header import DiskRadialModel, Disk2D, np, plt, MS, LS, au, kk, mp, finalize
import copy

opac    = ['supersimple', {'dusttogas': 0.01, 'kappadust': 1e2}]
disk    = DiskRadialModel(mdisk=0.01*MS,flang=0.05)
disk2d  = Disk2D(disk, zrmax=0.3, meanopacitymodel=opac)
disk2diso = copy.deepcopy(disk2d) 

for vert in disk2d.verts:
    vert.iterate_vertical_structure()

disk2diso.compute_vphi()
disk2d.compute_vphi()

ir = 80
vk = disk.omk[ir]*disk.r[ir]

plt.figure()
plt.plot(disk2d.cyl2d_zr[ir,:],disk2diso.cyl2d_vphi[ir,:]/vk,label='Isothermal disk')
plt.plot(disk2d.cyl2d_zr[ir,:],disk2d.cyl2d_vphi[ir,:]/vk,label='Irradiated disk')
plt.xlabel('z/r')
plt.ylabel(r'$v_\phi/v_K(z=0)$')
plt.legend()

finalize(results=(disk2d.cyl2d_vphi[ir,:]))
