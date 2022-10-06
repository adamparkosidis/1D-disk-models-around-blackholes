from snippet_header import DiskRadialModel, plt, au, finalize

flindex = 1.0   # Flaring index defined as Flaring angle = flindex * H_p/r
d = DiskRadialModel()
plt.plot(d.r / au, d.hp / d.r)
for iter in range(5):
    d.flang = flindex * d.hp / d.r
    d.compute_disktmid()
    plt.plot(d.r / au, d.hp / d.r)
plt.xscale('log')
plt.xlabel(r'$r [\mathrm{au}]$')
plt.ylabel(r'$H_p/r$')

finalize(results=(d.hp))
