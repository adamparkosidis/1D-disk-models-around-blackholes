from snippet_header import DiskRadialModel, np, plt, MS, au, finalize

d = DiskRadialModel(mdisk=0.01 * MS)
kappa = 1e3    # Take a somewhat low kappa for now
d.meanopacitymodel = ['supersimple', {'dusttogas': 0.01, 'kappadust': kappa}]
d.compute_mean_opacity()

plt.figure()
plt.plot(d.r / au, d.flang + np.zeros(len(d.r)))
for iter in range(10):
    d.compute_hsurf()
    d.compute_flareindex()
    d.compute_flareangle_from_flareindex(inclrstar=False)
    plt.plot(d.r / au, d.flang)
plt.xscale('log')
plt.xlabel('r [au]')
plt.ylabel(r'$\varphi$')
plt.xlim(right=200)

finalize(results=(d.flang))
