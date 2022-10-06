from snippet_header import DiskRadialModel, np, plt, MS, au, finalize

d = DiskRadialModel(mdisk=0.01 * MS)
kappa = 1e5
d.meanopacitymodel = ['supersimple', {'dusttogas': 0.01, 'kappadust': kappa}]
d.compute_mean_opacity()

plt.figure()
plt.plot(d.r / au, d.flang + np.zeros(len(d.r)))
iter1 = d.iterate_flaringangle(inclrstar=True, itermax=20, convcrit=1e-2, iterhp=False)
flangwithstar = d.flang
iter2 = d.iterate_flaringangle(inclrstar=False, itermax=20, convcrit=1e-2, iterhp=False)
flangpointstar = d.flang

plt.plot(d.r / au, flangwithstar, label='With star radius')
plt.plot(d.r / au, flangpointstar, label='Star as point source')
plt.xscale('log')
plt.xlim(right=200)
plt.xlabel('r [au]')
plt.ylabel(r'$\varphi$')
plt.legend(loc='lower right')
print('Nr of iterations = {}, {}'.format(iter1, iter2))

finalize(results=(flangwithstar,flangpointstar))
