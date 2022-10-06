from snippet_header import DiskRadialModel, Disk2D, np, plt, MS, LS, au, kk, mp, finalize

dradial = DiskRadialModel(mdisk=0.01*MS)
disk2d  = Disk2D(dradial)

disk2d.compute_vphi()

ir   = 80
cs2  = kk*disk2d.verts[ir].tgas/(2.3*mp)
vk   = disk2d.verts[ir].omk_full*disk2d.verts[ir].r
r    = disk2d.verts[ir].r
omk  = disk2d.verts[ir].omk_midpl
hp   = np.sqrt(cs2[0])/omk
hpr  = hp/r
p    = (np.log(disk2d.verts[ir+1].rhogas[0])-np.log(disk2d.verts[ir].rhogas[0]))/(np.log(disk2d.r[ir+1])-np.log(disk2d.r[ir]))
q    = (np.log(disk2d.verts[ir+1].tgas[0])-np.log(disk2d.verts[ir].tgas[0]))/(np.log(disk2d.r[ir+1])-np.log(disk2d.r[ir]))
zr   = disk2d.cyl2d_zr[ir,:]
#om0  = disk2d.cyl2d_vphi[ir,0]/r
#omana= om0 + (q*omk*zr**2/4.)   # Integral of Eq. 2 Lin & Youdin (2015), ApJ 811, 17
rsph = r*np.sqrt(1+zr**2)
omana= omk * np.sqrt((p+q)*hpr**2 + (1+q) - (q*r/rsph))   # Eq. 13 Nelson, Gressel & Umurhan (2013), MNRAS 435, 2610

plt.figure()
plt.plot(disk2d.cyl2d_zr[ir,:],disk2d.cyl2d_vphi[ir,:]/vk[0],label='Full numerical solution')
plt.plot(disk2d.cyl2d_zr[ir,:],omana/omk,':',label='Analytic solution')
plt.plot(disk2d.cyl2d_zr[ir,:],vk/vk[0],label='Kepler velocity')
plt.plot([0,hpr],[0.9,0.9],label=r'$h_p$',color='purple')
plt.plot([0,hpr],[0.9,0.9],'|',color='purple')
plt.xlabel('z/r')
plt.ylabel(r'$v_\phi/v_K(z=0)$')
plt.legend()

plt.figure()
rr       = disk2d.cyl2d_rr
omega    = disk2d.cyl2d_vphi/rr
lphi     = omega*rr**2
ephi     = (omega*rr)**2
vplevels = np.array([2.5,3.,3.5])*1e5 # Levels at velocities in km/s
rvpl     = np.interp(vplevels,disk2d.cyl2d_vphi[::-1,0],disk2d.r[::-1])
omlevels = (vplevels/rvpl)
lplevels = (vplevels*rvpl)[::-1]
eplevels = (vplevels**2)
plt.contour(disk2d.cyl2d_rr/au,disk2d.cyl2d_zr,omega,levels=omlevels,colors='red',linestyles='solid')
plt.contour(disk2d.cyl2d_rr/au,disk2d.cyl2d_zr,lphi,levels=lplevels,colors='green',linestyles='dashed')
plt.contour(disk2d.cyl2d_rr/au,disk2d.cyl2d_zr,ephi,levels=eplevels,colors='blue',linestyles='dotted')
plt.xlim(50,170)
plt.ylim(0,0.55)
plt.xlabel('r [au]')
plt.ylabel('z/r')
plt.title(r'$dotted=\Omega^2\,r_{cyl}^2,\;Solid=\Omega,\;dashed=\Omega\,r_{cyl}^2$')

finalize(results=(disk2d.cyl2d_vphi[ir,:]))
