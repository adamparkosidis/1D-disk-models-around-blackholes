from snippet_header import DiskRadialModel, plt, MS, au, finalize

d = DiskRadialModel(mdisk=0.01 * MS)
d.add_dust(agrain=1e-4)
kappa = 1e3
d.meanopacitymodel = ['belllin']
d.compute_mean_opacity()
d.compute_disktmid(vischeat=True)

plt.figure()
plt.plot(d.r / au, d.tmid, label='Mid')
plt.plot(d.r / au, d.tirr, label='Irradd')
plt.plot(d.r / au, d.tvisc, label='Viscous')
plt.xscale('log')
plt.yscale('log')
plt.xlim(right=200)
plt.xlabel('r [au]')
plt.ylabel(r'$T_{\mathrm{mid}}$')
plt.legend()

finalize(results=(d.tmid,d.tirr,d.tvisc))
