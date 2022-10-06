from snippet_header import plt, DiskRadialModel, MS, year, au, finalize

d = DiskRadialModel(rout=400 * au)
d.make_disk_from_lbp_alpha(1e-2 * MS, 1 * au, 1e-2, 1e5 * year)
d.compute_omega()
plt.plot(d.r / au, d.dvphi / 1e2)
plt.xscale('log')
plt.xlabel(r'$r$ [au]')
plt.ylabel(r'$(v_\phi-v_K) [\mathrm{m}/\mathrm{s}]$')

finalize(results=(d.dvphi))
