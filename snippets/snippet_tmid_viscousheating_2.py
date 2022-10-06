from snippet_header import DiskRadialModel, plt, MS, au, finalize

d = DiskRadialModel(mdisk=0.01 * MS)
d.add_dust(agrain=1e-4)
kappainfrared = 1e3
kappastellar  = 1e5
d.meanopacitymodel = ['supersimple', {'dusttogas': 0.01, 'kappadust': kappainfrared}]
d.compute_mean_opacity()
d.compute_disktmid(vischeat=True)
tirr0  = 1.0 * d.tirr
tvisc0 = 1.0 * d.tvisc
d.meanopacitymodel = ['supersimple', {'dusttogas': 0.01, 'kappadust': kappastellar}]
d.compute_mean_opacity()
d.iterate_flaringangle(inclrstar=True, itermax=20, convcrit=1e-2, iterhp=True, keeptvisc=True)
d.meanopacitymodel = ['supersimple', {'dusttogas': 0.01, 'kappadust': kappainfrared}]
d.compute_mean_opacity()
d.compute_disktmid(vischeat=True)

plt.figure()
plt.plot(d.r / au, tirr0, label='Irradd0')
plt.plot(d.r / au, d.tirr, label='Irradd')
plt.plot(d.r / au, tvisc0, label='Viscous0')
plt.plot(d.r / au, d.tvisc, label='Viscous')
plt.plot(d.r / au, d.tmid, label='Mid', linewidth=3)
plt.xscale('log')
plt.yscale('log')
plt.xlim(right=200)
plt.xlabel('r [au]')
plt.ylabel(r'$T_{\mathrm{mid}}$')
plt.legend()

finalize(results=(tirr0,tvisc0,d.tmid,d.tirr,d.tvisc))
