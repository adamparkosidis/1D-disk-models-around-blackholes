from snippet_header import DiskRadialModel, plt, MS, au, finalize

d = DiskRadialModel(mdisk=0.01 * MS)
kappa = 1e5
d.meanopacitymodel = ['supersimple', {'dusttogas': 0.01, 'kappadust': kappa}]
d.compute_mean_opacity()
d.iterate_flaringangle(inclrstar=True, itermax=20, convcrit=1e-2, iterhp=True)

plt.figure()
plt.plot(d.r / au, d.hs / d.r, label='surface height')
plt.plot(d.r / au, d.hp / d.r, label='pressure scale height')
plt.xscale('log')
plt.xlim(right=200)
plt.xlabel('r [au]')
plt.ylabel(r'$H/r$')
plt.legend(loc='upper left')

finalize(results=(d.hs,d.hp))
