from snippet_header import DiskRadialModel, au, MS, year, plt, finalize

a = DiskRadialModel(rout=1000 * au)
a.make_disk_from_lbp_alpha(1e-2 * MS, 1 * au, 1e-3, 1e6 * year)
a.compute_mdot_at_interfaces()
a.compute_vr_at_interfaces()

ri = a.get_ri()   # Get the interface radii

plt.figure()
plt.plot(ri / au, a.mdot / (MS / year))
plt.xscale('log')
plt.xlabel('r [au]')
plt.ylabel(r'$\dot M [M_\odot\mathrm{/year}]$')

plt.figure()
plt.plot(ri / au, a.vr)
plt.xscale('log')
plt.xlabel('r [au]')
plt.ylabel(r'$v_r [\mathrm{cm/s}]$')

finalize(results=(a.mdot,a.vr))
